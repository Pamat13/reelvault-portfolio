import en from "./en.json";
import es from "./es.json";

const translations = { en, es } as const;

export type Locale = keyof typeof translations;
export const defaultLocale: Locale = "en";

export function t(locale: Locale, key: string): string {
  const keys = key.split(".");
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let value: any = translations[locale] ?? translations[defaultLocale];
  for (const k of keys) {
    value = value?.[k];
  }
  return typeof value === "string" ? value : key;
}

const supportedLocales = Object.keys(translations) as Locale[];

export function getLocaleFromUrl(url: URL): Locale {
  const [, first] = url.pathname.split("/");
  return supportedLocales.includes(first as Locale)
    ? (first as Locale)
    : defaultLocale;
}
