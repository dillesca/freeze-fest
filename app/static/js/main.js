document.addEventListener("DOMContentLoaded", () => {
  const slides = document.querySelectorAll(".photo-slide");
  const prevBtn = document.querySelector(".carousel-btn--prev");
  const nextBtn = document.querySelector(".carousel-btn--next");
  let currentSlide = 0;
  let autoplayId;

  const showSlide = (index) => {
    slides.forEach((slide, idx) => {
      slide.classList.toggle("is-active", idx === index);
    });
    currentSlide = index;
  };

  const nextSlide = () => {
    if (slides.length === 0) return;
    const newIndex = (currentSlide + 1) % slides.length;
    showSlide(newIndex);
  };

  const prevSlide = () => {
    if (slides.length === 0) return;
    const newIndex = (currentSlide - 1 + slides.length) % slides.length;
    showSlide(newIndex);
  };

  const resetAutoplay = () => {
    if (autoplayId) {
      clearInterval(autoplayId);
    }
    autoplayId = setInterval(nextSlide, 5000);
  };

  if (slides.length > 0) {
    showSlide(0);
    resetAutoplay();
    nextBtn?.addEventListener("click", () => {
      nextSlide();
      resetAutoplay();
    });
    prevBtn?.addEventListener("click", () => {
      prevSlide();
      resetAutoplay();
    });
  }

  const fileInput = document.querySelector("#photo-upload");
  const fileNameLabel = document.querySelector("[data-file-name]");

  fileInput?.addEventListener("change", () => {
    if (!fileNameLabel) return;
    const files = fileInput.files;
    if (!files || files.length === 0) {
      fileNameLabel.textContent = "No file chosen";
      return;
    }
    fileNameLabel.textContent = files[0].name;
  });
});
