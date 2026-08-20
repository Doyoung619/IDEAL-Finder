BEGIN;

ALTER TABLE participants
    ADD COLUMN IF NOT EXISTS consent_version VARCHAR(32);

ALTER TABLE experiment_blocks
    ADD COLUMN IF NOT EXISTS strategy_parameters_json TEXT NOT NULL DEFAULT '{}';
ALTER TABLE experiment_blocks
    ADD COLUMN IF NOT EXISTS mu_data BYTEA;
ALTER TABLE experiment_blocks
    ADD COLUMN IF NOT EXISTS strategy_state_data BYTEA;
ALTER TABLE latent_images
    ADD COLUMN IF NOT EXISTS latent_data BYTEA;
ALTER TABLE latent_images
    ADD COLUMN IF NOT EXISTS image_data BYTEA;

CREATE TABLE IF NOT EXISTS experiment_sessions (
    session_id VARCHAR(36) PRIMARY KEY,
    participant_id VARCHAR(16) NOT NULL UNIQUE
        REFERENCES participants(participant_id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    persona_completed_at TIMESTAMPTZ NOT NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'persona_complete',
    schema_version VARCHAR(32) NOT NULL,
    app_version VARCHAR(64) NOT NULL,
    persona_condition JSONB NOT NULL,
    persona_data JSONB NOT NULL,
    theta_persona JSONB NOT NULL,
    experiment_seed BIGINT NOT NULL,
    algorithm_order JSONB NOT NULL,
    m_order JSONB NOT NULL,
    user_agent TEXT
);

CREATE INDEX IF NOT EXISTS ix_experiment_sessions_participant_id
    ON experiment_sessions(participant_id);
CREATE INDEX IF NOT EXISTS ix_experiment_sessions_status
    ON experiment_sessions(status);

ALTER TABLE experiment_blocks
    ADD COLUMN IF NOT EXISTS session_id VARCHAR(36)
        REFERENCES experiment_sessions(session_id);
CREATE INDEX IF NOT EXISTS ix_experiment_blocks_session_id
    ON experiment_blocks(session_id);

CREATE UNIQUE INDEX IF NOT EXISTS uq_selection_block_round
    ON selections(block_id, round_id);

CREATE TABLE IF NOT EXISTS experiment_rounds (
    round_id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36) NOT NULL REFERENCES experiment_sessions(session_id),
    block_id VARCHAR(64) NOT NULL REFERENCES experiment_blocks(block_id),
    round_index INTEGER NOT NULL,
    algorithm VARCHAR(32) NOT NULL,
    m_value INTEGER NOT NULL,
    status VARCHAR(24) NOT NULL DEFAULT 'submitted',
    started_at TIMESTAMPTZ,
    answered_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    response_time_ms INTEGER NOT NULL,
    query_points JSONB NOT NULL,
    final_query_points JSONB NOT NULL,
    generated_image_ids JSONB NOT NULL,
    selected_index INTEGER NOT NULL,
    selected_image_id VARCHAR(36) NOT NULL REFERENCES latent_images(image_id),
    selected_theta JSONB NOT NULL,
    ideal_similarity_rating INTEGER NOT NULL,
    choice_difficulty_rating INTEGER NOT NULL,
    beta_before DOUBLE PRECISION,
    beta_after DOUBLE PRECISION,
    posterior_mean JSONB,
    posterior_covariance JSONB,
    effective_sample_size DOUBLE PRECISION,
    expected_information_gain DOUBLE PRECISION,
    random_seed BIGINT NOT NULL,
    query_metadata JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ,
    CONSTRAINT uq_session_block_round
        UNIQUE (session_id, block_id, round_index)
);

CREATE INDEX IF NOT EXISTS ix_experiment_rounds_session_id
    ON experiment_rounds(session_id);
CREATE INDEX IF NOT EXISTS ix_experiment_rounds_block_id
    ON experiment_rounds(block_id);
CREATE INDEX IF NOT EXISTS ix_experiment_rounds_status
    ON experiment_rounds(status);

CREATE TABLE IF NOT EXISTS experiment_events (
    event_id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36) NOT NULL REFERENCES experiment_sessions(session_id),
    block_id VARCHAR(64) REFERENCES experiment_blocks(block_id),
    event_type VARCHAR(48) NOT NULL,
    event_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    payload JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_experiment_events_session_id
    ON experiment_events(session_id);
CREATE INDEX IF NOT EXISTS ix_experiment_events_block_id
    ON experiment_events(block_id);
CREATE INDEX IF NOT EXISTS ix_experiment_events_event_type
    ON experiment_events(event_type);

COMMIT;
