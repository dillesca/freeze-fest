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
});
