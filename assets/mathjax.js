window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    packages: { "[+]": ["ams", "noerrors", "color"] },
    tags: "none"
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  },
  loader: { load: ["[tex]/ams", "[tex]/noerrors", "[tex]/color"] }
};

document$.subscribe(() => {
  if (typeof MathJax !== "undefined" && MathJax.startup) {
    MathJax.startup.promise = MathJax.startup.promise
      .then(() => MathJax.typesetPromise())
      .catch(err => console.warn("MathJax typeset failed", err));
  }
});
