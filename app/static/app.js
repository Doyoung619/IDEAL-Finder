document.addEventListener("DOMContentLoaded", () => {
  const startedAt = performance.now();
  const reactionInput = document.querySelector("[data-reaction-input]");

  document.querySelectorAll("[data-selection-form]").forEach((form) => {
    const cards = [...form.querySelectorAll(".selectable-card")];
    const selectedInput = form.querySelector("[data-selected-input]");
    const submit = form.querySelector("[data-require-selection]");
    const reveal = form.querySelector("[data-reveal-evaluation]");
    const panel = form.querySelector("[data-evaluation-panel]");

    const refreshSubmit = () => {
      const hasSelection = Boolean(selectedInput?.value);
      if (reveal) reveal.disabled = !hasSelection;
      if (submit) submit.disabled = !hasSelection;
    };

    cards.forEach((card) => {
      card.addEventListener("click", () => {
        cards.forEach((item) => item.classList.remove("selected"));
        card.classList.add("selected");
        selectedInput.value = card.dataset.imageId;
        refreshSubmit();
      });
    });

    reveal?.addEventListener("click", () => {
      panel.hidden = false;
      panel.scrollIntoView({ behavior: "smooth", block: "center" });
    });

    form.addEventListener("submit", () => {
      if (reactionInput) {
        reactionInput.value = ((performance.now() - startedAt) / 1000).toFixed(3);
      }
    });
    refreshSubmit();
  });

  document.querySelectorAll("[data-chip-category]").forEach((category) => {
    const boxes = [...category.querySelectorAll('input[type="checkbox"]')];
    const countNode = category.querySelector("[data-chip-count]");
    const refresh = () => {
      const checked = boxes.filter((box) => box.checked);
      countNode.textContent = checked.length;
      boxes.forEach((box) => {
        box.disabled = checked.length >= 2 && !box.checked;
      });
    };
    boxes.forEach((box) => box.addEventListener("change", refresh));
    refresh();
  });

  document.querySelectorAll("[data-single-submit]").forEach((form) => {
    form.addEventListener("submit", () => {
      const overlay = document.querySelector("[data-loading-overlay]");
      const message = overlay?.querySelector("[data-loading-message]");
      if (message) {
        message.textContent = form.dataset.loadingMessage || "처리 중입니다…";
      }
      if (overlay) overlay.hidden = false;
      const buttons = form.querySelectorAll('button[type="submit"]');
      buttons.forEach((button) => {
        button.disabled = true;
        button.dataset.originalText = button.textContent;
        button.textContent = "저장 중…";
      });
    });
  });

  document.querySelectorAll("[data-persona-form]").forEach((form) => {
    const priorityBoxes = [...form.querySelectorAll('[name="priority"]')];
    const priorityCount = form.querySelector("[data-priority-count]");
    const refreshPriorities = () => {
      const selectedKeys = new Set(
        [...form.querySelectorAll('[data-persona-category] input:checked')]
          .filter((box) => box.value !== "no_preference")
          .map((box) => box.value)
      );
      priorityBoxes.forEach((box) => {
        const wrapper = box.closest("[data-priority-key]");
        const available = selectedKeys.has(box.value);
        wrapper.hidden = !available;
        if (!available) box.checked = false;
      });
      const checked = priorityBoxes.filter((box) => box.checked);
      priorityBoxes.forEach((box) => {
        box.disabled = !box.checked && checked.length >= 3;
      });
      if (priorityCount) priorityCount.textContent = checked.length;
    };
    form.querySelectorAll("[data-persona-category]").forEach((category) => {
      const boxes = [...category.querySelectorAll('input[type="checkbox"]')];
      const maximum = Number(category.dataset.max);
      const count = category.querySelector("[data-persona-count]");
      const refresh = (changed) => {
        const noPreference = boxes.find((box) => box.value === "no_preference");
        if (changed?.value === "no_preference" && changed.checked) {
          boxes.forEach((box) => { if (box !== changed) box.checked = false; });
        } else if (changed?.checked && noPreference) {
          noPreference.checked = false;
          const conflicts = new Set((changed.dataset.conflicts || "").split(",").filter(Boolean));
          boxes.forEach((box) => { if (conflicts.has(box.value)) box.checked = false; });
        }
        const concrete = boxes.filter((box) => box.checked && box.value !== "no_preference");
        boxes.forEach((box) => {
          if (box.value !== "no_preference") box.disabled = !box.checked && concrete.length >= maximum;
        });
        if (count) count.textContent = concrete.length;
        refreshPriorities();
      };
      boxes.forEach((box) => box.addEventListener("change", () => refresh(box)));
      refresh();
    });
    priorityBoxes.forEach((box) => box.addEventListener("change", refreshPriorities));
    refreshPriorities();
  });

});
