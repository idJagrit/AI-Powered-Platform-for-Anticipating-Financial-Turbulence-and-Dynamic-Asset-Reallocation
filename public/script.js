document.addEventListener("DOMContentLoaded", () => {
  document.querySelector(".btn-outline")?.addEventListener("click", () => {
    window.location.href = "signin";
  });

  document.querySelectorAll(".btn-filled").forEach((btn) => {
    btn.addEventListener("click", () => {
      window.location.href = "signup";
    });
  });

  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener("click", function (e) {
      e.preventDefault();
      const target = document.querySelector(this.getAttribute("href"));
      if (target) {
        target.scrollIntoView({ behavior: "smooth" });
      }
    });
  });

  document.querySelectorAll(".quick-actions button").forEach((btn) => {
    btn.addEventListener("click", () => {
      alert(`You clicked: ${btn.textContent}`);
    });
  });

  const profile = document.getElementById("profile");
  const dropdown = document.getElementById("dropdown");

  if (profile && dropdown) {
    profile.addEventListener("click", (e) => {
      e.stopPropagation();
      dropdown.style.display = dropdown.style.display === "block" ? "none" : "block";
    });

    document.addEventListener("click", (e) => {
      if (!profile.contains(e.target)) {
        dropdown.style.display = "none";
      }
    });
  }

  const signOutBtn = document.getElementById("signout-btn");
  if (signOutBtn) {
    signOutBtn.addEventListener("click", (e) => {
      e.preventDefault();
      window.location.href = "/signin";
    });
  }

  const assistant = document.getElementById("assistant");
  const assistantToggle = document.getElementById("ai-assistant-toggle");
  const assistantClose = document.getElementById("assistant-close");
  const assistantInput = assistant?.querySelector("input");

  if (assistant && assistantToggle) {
    assistantToggle.addEventListener("click", (e) => {
      e.preventDefault();

      if (assistant.style.display === "none" || getComputedStyle(assistant).display === "none") {
        assistant.style.display = "block";
        assistant.classList.add("expanded");
        assistant.classList.remove("collapsed");
        return;
      }

      const isExpanded = assistant.classList.contains("expanded");
      if (isExpanded) {
        assistant.classList.remove("expanded");
        assistant.classList.add("collapsed");
      } else {
        assistant.classList.add("expanded");
        assistant.classList.remove("collapsed");
      }
    });
  }

  if (assistant && assistantClose) {
    assistantClose.addEventListener("click", (e) => {
      e.stopPropagation();
      assistant.style.display = "none";
    });
  }

  if (assistantInput) {
    assistantInput.addEventListener("click", (e) => {
      e.stopPropagation();
    });
  }
});
