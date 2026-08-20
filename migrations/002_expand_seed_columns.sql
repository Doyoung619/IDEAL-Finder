BEGIN;

-- stable_seed returns an unsigned 32-bit value. PostgreSQL INTEGER is signed
-- 32-bit, so valid generated seeds above 2,147,483,647 must use BIGINT.
ALTER TABLE participants
    ALTER COLUMN base_seed TYPE BIGINT;
ALTER TABLE experiment_sessions
    ALTER COLUMN experiment_seed TYPE BIGINT;
ALTER TABLE experiment_blocks
    ALTER COLUMN initial_seed TYPE BIGINT;
ALTER TABLE latent_images
    ALTER COLUMN generator_seed TYPE BIGINT;
ALTER TABLE experiment_rounds
    ALTER COLUMN random_seed TYPE BIGINT;

COMMIT;
