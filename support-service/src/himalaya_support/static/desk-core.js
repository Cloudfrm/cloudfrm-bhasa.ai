/* desk-core.js — pure functions shared by the desk UI, the state harness and
   the Node test. No DOM access here. NFC only, never NFKC. */
(function (root, factory) {
  const api = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.DeskCore = api;
})(typeof window !== "undefined" ? window : globalThis, function () {
  "use strict";

  const DEVA_DIGITS = "०१२३४५६७८९";
  const DEVA_RE = /[ऀ-ॿ]/;

  function nfc(text) {
    return String(text || "").normalize("NFC");
  }

  /* ---------------------------------------------------------------- counts
     E4: both the overview stat and the inbox badge are derived from the same
     array by this one function, so they cannot diverge. */
  function deskCounts(state) {
    const threads = Array.isArray(state && state.threads) ? state.threads : [];
    const tickets = Array.isArray(state && state.tickets) ? state.tickets : [];
    const openTickets = tickets.filter((t) => t && t.status !== "resolved").length;
    return {
      threads: threads.length,
      inbox: threads.length, // same number, same source — by construction
      openTickets,
    };
  }

  /* -------------------------------------------------------------- numerals
     Rule (terminology.json): UI chrome digits follow the interface language.
     Quoted evidence is never passed through this. */
  function toDevaDigits(text) {
    return String(text).replace(/[0-9]/g, (d) => DEVA_DIGITS[Number(d)]);
  }
  function fmtNum(n, lang) {
    const s = String(n);
    return lang === "ne" ? toDevaDigits(s) : s;
  }

  /* ------------------------------------------------------------------ time
     Relative time for lists; 24-hour clock for absolute times; Gregorian
     labelled when a date is shown. */
  function relativeTime(iso, lang, now) {
    const then = new Date(iso);
    if (isNaN(then.getTime())) return "";
    const ms = (now ? new Date(now) : new Date()).getTime() - then.getTime();
    const sec = Math.max(0, Math.round(ms / 1000));
    const min = Math.round(sec / 60);
    const hr = Math.round(min / 60);
    const day = Math.round(hr / 24);
    const ne = lang === "ne";
    let out;
    if (sec < 45) out = ne ? "भर्खरै" : "just now";
    else if (min < 60) out = ne ? `${min} मिनेट अघि` : `${min} min ago`;
    else if (hr < 24) out = ne ? `${hr} घण्टा अघि` : `${hr} h ago`;
    else if (day < 7) out = ne ? `${day} दिन अघि` : `${day} d ago`;
    else out = absoluteDate(iso, lang);
    return ne ? toDevaDigits(out) : out;
  }

  function pad(n) { return String(n).padStart(2, "0"); }

  function clock24(iso, lang) {
    const d = new Date(iso);
    if (isNaN(d.getTime())) return "";
    const s = `${pad(d.getHours())}:${pad(d.getMinutes())}`;
    return lang === "ne" ? toDevaDigits(s) : s;
  }

  function absoluteDate(iso, lang) {
    const d = new Date(iso);
    if (isNaN(d.getTime())) return "";
    const s = `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
    // Gregorian, and labelled as such (calendar rule).
    return lang === "ne" ? `${toDevaDigits(s)} ई.सं.` : `${s} AD`;
  }

  /* --------------------------------------------------------- credentials
     Client-side pre-send warning. The server is the authority; this only
     stops the obvious cases before they leave the composer. */
  const CARD = /(?<![\d\w])(?:\d[ -]?){13,19}(?![\d\w])/;
  const CVV = /\b(?:cvv|cvc|cvv2|सीभीभी)\s*[:=\-]?\s*\d{3,4}\b/i;
  const PASSWORD = /(?:\b(?:password|passwd|pwd|pass|पासवर्ड)\b|पासवर्ड)\s*(?:is|:|=|-|हो|चाहिँ)?\s*["“']?[^\s"”']{4,}/i;
  const PIN_WORD = /\b(?:pin|otp|mpin|tpin|पिन|ओटीपी|ओटिपी)\b/i;
  const BARE = /(?<![\d.,/-])\d{4,8}(?![\d.,/-])/g;
  const UNIT_AFTER = /^\s*(?:%|प्रतिशत|percent|रुपैयाँ|रुपियाँ|रु\.?|rs\.?|rupees?|paisa|पैसा|npr|usd|dollars?|मिनेट|minutes?|mins?|दिन|days?|घण्टा|hours?|hrs?|बजे|वर्ष|years?|पटक|times|अङ्क|digits?|अक्षर|characters?|महिना|months?)/i;
  const UNIT_BEFORE = /(?:रु\.?|रुपैयाँ|rs\.?|npr|usd|\$|₹|#|no\.?|नं\.?|number|नम्बर)\s*$/i;

  function credentialShapes(text) {
    let work = String(text || "").replace(/[०-९]/g, (d) => String(DEVA_DIGITS.indexOf(d)));
    const kinds = new Set();
    const blank = (re, kind) => {
      const g = new RegExp(re.source, re.flags.includes("g") ? re.flags : re.flags + "g");
      work = work.replace(g, (s) => { kinds.add(kind); return " ".repeat(s.length); });
    };
    blank(CARD, "card");
    blank(CVV, "cvv");
    blank(PASSWORD, "password");
    const pinWord = PIN_WORD.test(work);
    let m;
    BARE.lastIndex = 0;
    while ((m = BARE.exec(work))) {
      const after = work.slice(m.index + m[0].length, m.index + m[0].length + 14);
      const before = work.slice(Math.max(0, m.index - 12), m.index);
      if (UNIT_AFTER.test(after) || UNIT_BEFORE.test(before)) continue;
      if (pinWord || m[0].length === 4 || m[0].length === 6) kinds.add("pin_otp");
    }
    return Array.from(kinds);
  }

  /* ---------------------------------------------------------- word runs
     Mirror of the server segmenter, used to find the run under the caret and
     to decide locally whether a run can ever be converted. */
  const PROTECTED_ANY_CASE = new Set(["nimb", "scb", "usd", "npr", "swift", "emi", "ipo", "cib", "qr", "sms", "cvv", "ips", "connectips", "esewa", "khalti", "imepay", "fonepay", "nrb", "nepse", "vat", "nchl", "rtgs", "neft", "inr", "eur", "gbp", "aud"]);

  function protectedReason(run) {
    if (!run) return "empty";
    if (DEVA_RE.test(run)) return "devanagari";
    if (/^(?:https?:\/\/|www\.)/i.test(run)) return "url";
    if (/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(run)) return "email";
    if (/\d/.test(run)) return "digits";
    if (PROTECTED_ANY_CASE.has(run.toLowerCase())) return "protected_term";
    const letters = run.replace(/[^A-Za-z]/g, "");
    if (letters.length >= 2 && letters === letters.toUpperCase()) return "all_caps";
    return null;
  }

  /* The Latin run that ends exactly at `caret`, or null. */
  function runAtCaret(value, caret) {
    const head = String(value || "").slice(0, caret);
    const m = head.match(/[A-Za-z][A-Za-z']*$/);
    if (!m) return null;
    const start = caret - m[0].length;
    // A run glued to digits/@/: is part of something protected.
    const before = head.slice(0, start);
    if (/[\d@:\/.]$/.test(before)) return null;
    const after = String(value || "").slice(caret);
    if (/^[A-Za-z0-9@.]/.test(after)) return null; // caret is inside a token, not at its end
    return { text: m[0], start, end: caret };
  }

  function detectLanguage(text) {
    // Devanagari vs plain Latin after digits/urls/emails/ALL-CAPS are set aside.
    let deva = 0, latin = 0;
    for (const tok of String(text || "").split(/\s+/)) {
      if (!tok) continue;
      if (protectedReason(tok) && !DEVA_RE.test(tok)) continue;
      if (DEVA_RE.test(tok)) deva += tok.length; else if (/[A-Za-z]/.test(tok)) latin += tok.length;
    }
    return deva >= latin ? "ne" : "en";
  }

  return { nfc, deskCounts, toDevaDigits, fmtNum, relativeTime, clock24, absoluteDate, credentialShapes, protectedReason, runAtCaret, detectLanguage };
});
