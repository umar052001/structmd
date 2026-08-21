/* structmd docs — interactions */
(function () {
  "use strict";

  /* ------------------------------------------------------------------
     Theme toggle (persisted in localStorage, defaults to system)
  ------------------------------------------------------------------ */
  var root = document.documentElement;
  var stored = null;
  try { stored = localStorage.getItem("structmd-theme"); } catch (e) { /* private mode */ }

  function applyTheme(theme) {
    root.setAttribute("data-theme", theme);
    try { localStorage.setItem("structmd-theme", theme); } catch (e) { /* ignore */ }
  }

  if (!stored) {
    var prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
    applyTheme(prefersDark ? "dark" : "light");
  } else {
    applyTheme(stored);
  }

  document.querySelectorAll(".theme-toggle").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var current = root.getAttribute("data-theme") === "dark" ? "dark" : "light";
      applyTheme(current === "dark" ? "light" : "dark");
    });
  });

  /* ------------------------------------------------------------------
     Mobile navigation
  ------------------------------------------------------------------ */
  var menuBtn = document.querySelector(".menu-btn");
  var navLinks = document.querySelector(".nav-links");
  if (menuBtn && navLinks) {
    menuBtn.addEventListener("click", function () {
      navLinks.classList.toggle("open");
    });
    navLinks.querySelectorAll("a").forEach(function (a) {
      a.addEventListener("click", function () { navLinks.classList.remove("open"); });
    });
  }

  /* ------------------------------------------------------------------
     Copy-to-clipboard buttons.
     A button copies either data-copy text or its sibling <pre> content.
  ------------------------------------------------------------------ */
  document.querySelectorAll(".copy-btn").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var target = btn.getAttribute("data-copy");
      var text = "";
      if (target) {
        var el = document.querySelector(target);
        text = el ? el.textContent : "";
      } else {
        var pre = btn.parentElement.querySelector("pre");
        text = pre ? pre.textContent : "";
      }
      if (!text) return;

      function done() {
        btn.classList.add("copied");
        var original = btn.getAttribute("aria-label") || "Copy";
        btn.setAttribute("aria-label", "Copied!");
        setTimeout(function () {
          btn.classList.remove("copied");
          btn.setAttribute("aria-label", original);
        }, 1400);
      }

      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, done);
      } else {
        var ta = document.createElement("textarea");
        ta.value = text;
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand("copy"); } catch (e) { /* ignore */ }
        document.body.removeChild(ta);
        done();
      }
    });
  });

  /* ------------------------------------------------------------------
     Tabs (quick start panels)
  ------------------------------------------------------------------ */
  document.querySelectorAll("[data-tabs]").forEach(function (group) {
    var tabs = group.querySelectorAll(".tab");
    var scope = group.getAttribute("data-tabs");
    tabs.forEach(function (tab) {
      tab.addEventListener("click", function () {
        tabs.forEach(function (t) { t.classList.remove("active"); });
        tab.classList.add("active");
        document.querySelectorAll('.panel[data-panel-group="' + scope + '"]').forEach(function (p) {
          p.classList.toggle("active", p.getAttribute("data-panel") === tab.getAttribute("data-target"));
        });
      });
    });
  });

  /* ------------------------------------------------------------------
     Scrollspy for docs sidebar
  ------------------------------------------------------------------ */
  var sideLinks = Array.prototype.slice.call(document.querySelectorAll(".sidebar a.side-link"));
  if (sideLinks.length && "IntersectionObserver" in window) {
    var sections = sideLinks
      .map(function (a) {
        var id = a.getAttribute("href").split("#")[1];
        return id ? document.getElementById(id) : null;
      })
      .filter(Boolean);

    var visible = new Map();
    var observer = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          visible.set(entry.target.id, entry.isIntersecting ? entry.intersectionRatio : 0);
        });
        var best = null;
        var bestRatio = 0;
        visible.forEach(function (ratio, id) {
          if (ratio > bestRatio) { bestRatio = ratio; best = id; }
        });
        if (best) {
          sideLinks.forEach(function (a) {
            a.classList.toggle("active", a.getAttribute("href").indexOf(best) !== -1);
          });
        }
      },
      { rootMargin: "-15% 0px -60% 0px", threshold: [0, 0.25, 0.5] }
    );
    sections.forEach(function (s) { observer.observe(s); });
  }

  /* ------------------------------------------------------------------
     Anchor links inside headings
  ------------------------------------------------------------------ */
  document.querySelectorAll(".doc-content h2[id], .doc-content h3[id]").forEach(function (h) {
    if (h.querySelector(".anchor-link")) return;
    var a = document.createElement("a");
    a.className = "anchor-link";
    a.href = "#" + h.id;
    a.setAttribute("aria-label", "Link to this section");
    a.textContent = "#";
    h.appendChild(a);
  });
})();
