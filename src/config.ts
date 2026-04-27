export const SITE = {
  website: "https://example.com", // TODO: replace with deployed domain
  author: "Jorge Pérez Amat",
  profile: "https://github.com/Pamat13",
  desc: "Math grad, data + ML. Building local-first AI tools.",
  title: "Jorge Pérez Amat",
  ogImage: "astropaper-og.jpg",
  lightAndDarkMode: true,
  postPerIndex: 4,
  postPerPage: 4,
  scheduledPostMargin: 15 * 60 * 1000, // 15 minutes
  showArchives: true,
  showBackButton: true, // show back button in post detail
  editPost: {
    enabled: false,
    text: "Edit page",
    url: "https://github.com/Pamat13/reelvault-portfolio/edit/main/",
  },
  dynamicOgImage: true,
  dir: "ltr", // "rtl" | "auto"
  lang: "en", // html lang code. Set this empty and default will be "en"
  timezone: "Europe/Madrid", // Default global timezone (IANA format) https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
} as const;
