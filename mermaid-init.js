// Tự động nhận dạng navy theme hoặc light theme để đổi màu nền mermaid phù hợp
(function() {
  var isDark = document.querySelector('html').classList.contains('navy') || document.querySelector('html').classList.contains('coal');
  mermaid.initialize({
    startOnLoad: true,
    theme: isDark ? 'dark' : 'default',
    themeVariables: {
      primaryColor: isDark ? '#1f2937' : '#f3f4f6',
      primaryTextColor: isDark ? '#f3f4f6' : '#111827',
      primaryBorderColor: isDark ? '#374151' : '#e5e7eb',
      lineColor: isDark ? '#9ca3af' : '#4b5563',
      secondaryColor: isDark ? '#111827' : '#ffffff',
      tertiaryColor: isDark ? '#374151' : '#f9fafb'
    }
  });
})();
