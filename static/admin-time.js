(function (root) {
  'use strict';
  const formatter = new Intl.DateTimeFormat('fr-FR', {
    timeZone: 'Europe/Paris', year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', hourCycle: 'h23'
  });
  function format(value) {
    if (!value || typeof value !== 'string') return 'Date non renseignée';
    // Legacy records were written on Render with TZ=Europe/Paris. Do not
    // reinterpret these wall-clock values as UTC or the viewer's timezone.
    const legacy = /^(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2})(?::\d{2})?$/.exec(value);
    if (legacy) {
      const [, year, month, day, hour, minute] = legacy;
      const date = new Date(Date.UTC(+year, +month - 1, +day));
      if (date.getUTCFullYear() !== +year || date.getUTCMonth() !== +month - 1 ||
          date.getUTCDate() !== +day || +hour > 23 || +minute > 59) return 'Date à vérifier';
      return `${day}/${month}/${year} à ${hour}:${minute} (Paris)`;
    }
    if (!/^\d{4}-\d{2}-\d{2}T.*(?:Z|[+-]\d{2}:\d{2})$/.test(value)) return 'Date à vérifier';
    const date = new Date(value);
    if (!Number.isFinite(date.getTime())) return 'Date à vérifier';
    return formatter.format(date) + ' (Paris)';
  }
  root.RmgAdminTime = { format };
  if (typeof module !== 'undefined' && module.exports) module.exports = { format };
})(typeof window === 'undefined' ? globalThis : window);
