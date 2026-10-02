/* Shared by the policy pages (contact, terms, privacy, refund-policy).
   Same mobile-nav toggle and page-transition curtain as the home page. */

(() => {
    const navbar = document.querySelector(".navbar");
    const btn = document.querySelector(".menu-btn");
    if (!navbar || !btn) return;
    const setOpen = (open) => {
        navbar.classList.toggle("open", open);
        btn.setAttribute("aria-expanded", String(open));
        btn.setAttribute("aria-label", open ? "Close menu" : "Open menu");
    };
    btn.addEventListener("click", () => setOpen(!navbar.classList.contains("open")));
    document.querySelectorAll(".nav-link").forEach((l) => l.addEventListener("click", () => setOpen(false)));
    window.addEventListener("resize", () => { if (window.innerWidth > 768) setOpen(false); });
})();

(() => {
    const transition = document.querySelector(".page-transition");
    if (!transition) return;

    document.querySelectorAll("a").forEach((link) => {
        link.addEventListener("click", (e) => {
            const href = link.getAttribute("href");

            /* New-tab links, mailto and ctrl/cmd-clicks keep their normal behaviour. */
            if (link.target === "_blank" || e.ctrlKey || e.metaKey || e.shiftKey || e.button !== 0) return;
            if (!href || href.startsWith("#") || href.startsWith("mailto:")) return;

            e.preventDefault();
            transition.classList.add("active");
            setTimeout(() => {
                window.location.href = href;
            }, 500);
        });
    });

    window.addEventListener("pageshow", () => transition.classList.remove("active"));
})();
