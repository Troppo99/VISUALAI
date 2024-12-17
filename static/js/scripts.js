// static/js/scripts.js
// Toggle class active untuk hamburger menu
const navbarNav = document.querySelector(".navbar-nav");
const hamburger = document.querySelector("#hamburger-report"); // Sesuaikan ID

hamburger.onclick = () => {
  navbarNav.classList.toggle("active");
};

// Klik untuk hide sidebar
document.addEventListener("click", function (e) {
  if (!hamburger.contains(e.target) && !navbarNav.contains(e.target)) {
    navbarNav.classList.remove("active");
  }
});

// Fullscreen functionality for multiple videos
document.addEventListener("DOMContentLoaded", () => { // Pastikan DOM sudah siap
  const fullscreenButtons = document.querySelectorAll(".fullscreen-btn");

  fullscreenButtons.forEach(button => {
    button.addEventListener("click", () => {
      const videoContainer = button.closest(".video-container");

      if (!document.fullscreenElement) {
        // Meminta fullscreen pada videoContainer
        if (videoContainer.requestFullscreen) {
          videoContainer.requestFullscreen();
        } else if (videoContainer.webkitRequestFullscreen) { /* Safari */
          videoContainer.webkitRequestFullscreen();
        } else if (videoContainer.msRequestFullscreen) { /* IE11 */
          videoContainer.msRequestFullscreen();
        }

        // Ubah ikon ke minimize saat fullscreen
        button.querySelector("i").setAttribute("data-feather", "minimize");
        feather.replace(); // Render ulang ikon
      } else {
        // Keluar dari fullscreen
        if (document.exitFullscreen) {
          document.exitFullscreen();
        }

        // Kembali ke ikon maximize saat keluar fullscreen
        button.querySelector("i").setAttribute("data-feather", "maximize");
        feather.replace(); // Render ulang ikon
      }
    });
  });

  // Listener untuk perubahan fullscreen dari sumber lain (misalnya, tombol Esc)
  document.addEventListener("fullscreenchange", () => {
    fullscreenButtons.forEach(button => {
      if (!document.fullscreenElement) {
        button.querySelector("i").setAttribute("data-feather", "maximize");
        feather.replace(); // Render ulang ikon
      }
    });
  });
});
