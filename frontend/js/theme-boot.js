(function () {
      try {
        var saved = localStorage.getItem("bhasa-theme");
        var dark = saved ? saved === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
        document.documentElement.classList.toggle("dark", dark);
        document.documentElement.style.colorScheme = dark ? "dark" : "light";
      } catch (e) {}
    })();
