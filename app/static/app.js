document.addEventListener("DOMContentLoaded", () => {
  const startedAt = performance.now();
  const reactionInput = document.querySelector("[data-reaction-input]");
  const loadingScreen = document.querySelector("[data-loading-screen]");

  const showLoading = () => {
    if (!loadingScreen) return;
    loadingScreen.hidden = false;
    requestAnimationFrame(() => loadingScreen.classList.add("is-visible"));
  };

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
      showLoading();
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
      const buttons = form.querySelectorAll('button[type="submit"]');
      buttons.forEach((button) => {
        button.disabled = true;
        button.dataset.originalText = button.textContent;
        button.textContent = "저장 중…";
      });
      showLoading();
    });
  });

});
