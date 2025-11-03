document.addEventListener("DOMContentLoaded", () => {
  const setupCarousel = (carousel) => {
    const slides = carousel.querySelectorAll(".photo-slide");
    if (slides.length === 0) {
      return;
    }

    const prevBtn = carousel.querySelector(".carousel-btn--prev");
    const nextBtn = carousel.querySelector(".carousel-btn--next");
    let currentSlide = Array.from(slides).findIndex((slide) => slide.classList.contains("is-active"));
    if (currentSlide < 0) {
      currentSlide = 0;
    }
    let autoplayId;

    const showSlide = (index) => {
      slides.forEach((slide, idx) => {
        slide.classList.toggle("is-active", idx === index);
      });
      currentSlide = index;
    };

    const nextSlide = () => {
      if (slides.length <= 1) return;
      const newIndex = (currentSlide + 1) % slides.length;
      showSlide(newIndex);
    };

    const prevSlide = () => {
      if (slides.length <= 1) return;
      const newIndex = (currentSlide - 1 + slides.length) % slides.length;
      showSlide(newIndex);
    };

    const resetAutoplay = () => {
      if (autoplayId) {
        clearInterval(autoplayId);
      }
      if (slides.length > 1) {
        autoplayId = setInterval(nextSlide, 5000);
      }
    };

    showSlide(currentSlide);
    resetAutoplay();

    nextBtn?.addEventListener("click", () => {
      nextSlide();
      resetAutoplay();
    });
    prevBtn?.addEventListener("click", () => {
      prevSlide();
      resetAutoplay();
    });
  };

  document.querySelectorAll(".photo-carousel").forEach(setupCarousel);

  const fileInput = document.querySelector("#photo-upload");
  const fileNameLabel = document.querySelector("[data-file-name]");

  fileInput?.addEventListener("change", () => {
    if (!fileNameLabel) return;
    const files = fileInput.files;
    if (!files || files.length === 0) {
      fileNameLabel.textContent = "No files chosen";
      return;
    }
    if (files.length === 1) {
      fileNameLabel.textContent = files[0].name;
    } else {
      fileNameLabel.textContent = `${files.length} files selected`;
    }
  });

  document.querySelectorAll("[data-rsvp-cancel]").forEach((button) => {
    button.addEventListener("click", () => {
      const form = button.closest("form");
      const details = button.closest("details");
      form?.reset();
      if (details) {
        details.open = false;
      }
    });
  });

  const initLightbox = () => {
    const triggers = Array.from(document.querySelectorAll("[data-lightbox-src]"));
    if (triggers.length === 0) {
      return;
    }

    let lastFocused = null;

    const overlay = document.createElement("div");
    overlay.className = "lightbox-overlay";
    overlay.innerHTML = `
      <div class="lightbox-overlay__backdrop" data-lightbox-close></div>
      <figure class="lightbox-overlay__content" role="dialog" aria-modal="true" aria-label="Expanded photo view">
        <button type="button" class="lightbox-overlay__close" aria-label="Close photo" data-lightbox-close>&times;</button>
        <img class="lightbox-overlay__image" src="" alt="" data-lightbox-image />
        <div class="lightbox-overlay__actions">
          <button type="button" class="lightbox-overlay__download" data-lightbox-download>Download</button>
        </div>
        <figcaption class="lightbox-overlay__caption is-hidden" data-lightbox-caption></figcaption>
      </figure>
    `;
    document.body.appendChild(overlay);

    const imageEl = overlay.querySelector("[data-lightbox-image]");
    const captionEl = overlay.querySelector("[data-lightbox-caption]");
    const closeControls = overlay.querySelectorAll("[data-lightbox-close]");
    const closeButton = overlay.querySelector(".lightbox-overlay__close");

    const downloadButton = overlay.querySelector("[data-lightbox-download]");
    if (downloadButton) {
      downloadButton.setAttribute("disabled", "true");
    }
    let currentSrc = null;

    const close = () => {
      overlay.classList.remove("is-visible");
      document.body.classList.remove("lightbox-open");
      imageEl.src = "";
      imageEl.alt = "";
      captionEl.textContent = "";
      captionEl.classList.add("is-hidden");
      downloadButton?.setAttribute("disabled", "true");
      currentSrc = null;
      if (lastFocused && typeof lastFocused.focus === "function") {
        lastFocused.focus();
      }
      lastFocused = null;
    };

    const open = (src, label, altText) => {
      if (!src) return;
      imageEl.src = src;
      const safeLabel = label || "Freeze Fest photo";
      imageEl.alt = altText || safeLabel;
      captionEl.textContent = "";
      captionEl.classList.add("is-hidden");
      currentSrc = src;
      if (downloadButton) {
        downloadButton.disabled = false;
      }
      overlay.classList.add("is-visible");
      document.body.classList.add("lightbox-open");
      window.requestAnimationFrame(() => {
        closeButton?.focus();
      });
    };

    triggers.forEach((trigger) => {
      trigger.addEventListener("click", (event) => {
        event.preventDefault();
        const src = trigger.getAttribute("data-lightbox-src") || trigger.getAttribute("src");
        const label = trigger.getAttribute("data-lightbox-label") || trigger.getAttribute("alt");
        const altText = trigger.getAttribute("alt");
        lastFocused = trigger;
        open(src, label, altText);
      });
    });

    closeControls.forEach((control) => {
      control.addEventListener("click", (event) => {
        event.preventDefault();
        close();
      });
    });

    overlay.addEventListener("click", (event) => {
      if (event.target === overlay || event.target.classList.contains("lightbox-overlay__backdrop")) {
        close();
      }
    });

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && overlay.classList.contains("is-visible")) {
        close();
      }
    });

    const mimeExtensionMap = {
      "image/jpeg": "jpg",
      "image/png": "png",
      "image/gif": "gif",
      "image/webp": "webp",
      "image/heic": "heic",
      "image/heif": "heif",
    };

    const deriveExtension = (src, blobType) => {
      if (blobType && mimeExtensionMap[blobType]) {
        return mimeExtensionMap[blobType];
      }
      if (typeof src !== "string") {
        return "";
      }
      const cleanSrc = src.split(/[?#]/)[0];
      const parts = cleanSrc.split(".");
      if (parts.length > 1) {
        const candidate = parts.pop();
        if (candidate) {
          return candidate.toLowerCase();
        }
      }
      return "";
    };

    downloadButton?.addEventListener("click", async () => {
      if (!currentSrc) {
        return;
      }
      try {
        downloadButton.disabled = true;
        const response = await fetch(currentSrc, { mode: "cors", credentials: "omit" });
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const blob = await response.blob();
        const extension = deriveExtension(currentSrc, blob.type);
        const fileName = `freeze_fest_${Date.now()}${extension ? "." + extension : ""}`;
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = fileName;
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
      } catch (error) {
        console.warn("Unable to download photo", error);
        window.alert("We couldn't download that photo automatically. Please try again or save it manually.");
      } finally {
        if (overlay.classList.contains("is-visible")) {
          downloadButton.disabled = false;
        }
      }
    });
  };

  initLightbox();
});
