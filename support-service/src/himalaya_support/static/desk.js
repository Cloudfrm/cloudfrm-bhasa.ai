/* desk.js — officer desk application. The intended user is the bank officer:
   they type the member's question; bhasa quotes the matching passage or
   refuses. Members never see this surface. */
(function () {
  "use strict";
  const core = window.DeskCore;
  const ui = window.DeskUI;

  const CHROME = {
    ne: {
      title: "bhasa · अधिकृत डेस्क",
      brandSub: "अधिकृत डेस्क · CloudFRM",
      home: "डेसबोर्ड", chat: "च्याट", settings: "सेटिङ", call: "कल",
      pageHome: "अवलोकन", pageChat: "सदस्य सहयोग", pageSettings: "सेटिङ", pageCall: "कल",
      statusOpen: (t) => `सेवा खुला · जाँच ${t}`,
      statusDown: (t) => `सेवा पुग्न सकिएन · अन्तिम जाँच ${t}`,
      statusChecking: "सेवा जाँच हुँदै…",
      langGroup: "इन्टरफेस भाषा",
      themeToLight: "उज्यालो", themeToDark: "अँध्यारो", themeToggle: "उज्यालो र अँध्यारो मोड बदल्नुहोस्",
      homeTitle: "अधिकृत डेस्क",
      homeLede: "सदस्यको प्रश्न लेख्नुहोस्। bhasa ले लोड गरिएका बैंक/उत्पादन दस्तावेजबाट मिल्ने अंश खोज्छ, त्यही अंशका आधारमा भाषा-मोडेल (Gemma) ले जवाफ लेख्छ, र प्रमाणका रूपमा अंश जस्ताको तस्तै देखाउँछ। दस्तावेजमा नभएको अंक जवाफमा आए, वा मिल्ने अंश नभेटिए, स्थिर अस्वीकार वाक्य आउँछ। मोडेल पुग्न नसके अंश मात्र देखिन्छ।",
      statChat: "च्याट संवाद", statTicket: "खुला टिकट", statDocs: "लोड गरिएका दस्तावेज",
      doorChatKicker: "लेखेर", doorChatTitle: "सदस्य सहयोग च्याट",
      doorChatBody: "सदस्यको प्रश्न लेख्नुहोस्। रोमन अक्षर शब्द-शब्द गरी देवनागरीमा बदल्न सकिन्छ; अंग्रेजी र संकेत (KYC, OTP, NPR…) जस्ताको तस्तै रहन्छन्।",
      doorVoiceKicker: "बोलेर", doorVoiceTitle: "फोन लाइन पछि आउँछ",
      doorVoiceBody: (t) => `आवाज सेवा (STT/TTS) अहिले तैनाथ छैन — /capabilities ले जाँच ${t} मा यसै भन्छ। तैनाथ भएपछि मात्र यहाँ नियन्त्रण देखिन्छ।`,
      inbox: "संवादहरू", newChat: "+ नयाँ संवाद", memberRow: "सदस्य", tickets: "टिकट",
      emptyInbox: "अहिले संवाद छैन", emptyTickets: "खुला टिकट छैन",
      topics: "नमूना प्रश्न — दस्तावेजबाट जवाफ आउने पुष्टि भएका",
      topicsNone: "अहिले पुष्टि भएको नमूना प्रश्न छैन",
      roomNew: "नयाँ संवाद", roomNewSub: "सदस्यले सोधेको प्रश्न लेख्नुहोस्, वा तलको नमूना प्रश्न छान्नुहोस्।",
      roomThread: "संवाद", roomMsgs: "सन्देश", roomSince: "सुरु", roomTag: "दस्तावेजमा आधारित जवाफ · प्रमाण जस्ताको तस्तै",
      placeholder: "सदस्यको प्रश्न यहाँ लेख्नुहोस्…",
      hint: "Enter ले पठाउँछ · Shift+Enter ले नयाँ लाइन",
      send: "पठाउनुहोस्",
      settingsTitle: "सेटिङ",
      settingsBody: "गोप्य कुञ्जीहरू यो स्क्रिनमा कहिल्यै राखिँदैनन् — ती सर्भरको सुरक्षित सेटिङ र प्रदायकको ड्यासबोर्डमा मात्र बस्छन्। तल सेवाको वास्तविक क्षमता र यस डेस्कका नियम देखिन्छन्।",
      capsTitle: "क्षमता (/v1/capabilities)", rulesTitle: "शब्दावली र ढाँचा नियम",
      yes: "उपलब्ध", no: "तैनाथ छैन",
      capRows: { answer_path: "जवाफको बाटो", llm: "भाषा-मोडेल", documents: "दस्तावेज", stt: "आवाज → पाठ (STT)", tts: "पाठ → आवाज (TTS)", grounding: "अस्वीकार वाक्य", rate_limit: "दर सीमा", checked_at: "जाँच" },
      llmUnreachable: "पुग्न सकिएन — दस्तावेजको अंश जस्ताको तस्तै देखाइन्छ",
      mic: "बोल्नुहोस्",
    },
    en: {
      title: "bhasa · Officer desk",
      brandSub: "Officer desk · CloudFRM",
      home: "Dashboard", chat: "Chat", settings: "Settings", call: "Call",
      pageHome: "Overview", pageChat: "Member support", pageSettings: "Settings", pageCall: "Call",
      statusOpen: (t) => `Service open · checked ${t}`,
      statusDown: (t) => `Service unreachable · last check ${t}`,
      statusChecking: "Checking service…",
      langGroup: "Interface language",
      themeToLight: "Light", themeToDark: "Dark", themeToggle: "Toggle light and dark theme",
      homeTitle: "Officer desk",
      homeLede: "Type the member's question. bhasa retrieves the matching passage from the loaded bank/product documents, a language model (Gemma) writes the reply from that passage, and the passage is shown verbatim as evidence. If the reply contains a figure the document does not, or nothing matches, the fixed refusal sentence is returned. If the model is unreachable, only the passage is shown.",
      statChat: "Chat threads", statTicket: "Open tickets", statDocs: "Documents loaded",
      doorChatKicker: "Type", doorChatTitle: "Member support chat",
      doorChatBody: "Type the member's question. Roman letters can be converted to Devanagari word by word; English and codes (KYC, OTP, NPR…) stay exactly as typed.",
      doorVoiceKicker: "Speak", doorVoiceTitle: "A phone line comes later",
      doorVoiceBody: (t) => `Voice (STT/TTS) is not deployed — /capabilities said so at ${t}. Controls appear here only once it is.`,
      inbox: "Conversations", newChat: "+ New chat", memberRow: "Member", tickets: "Tickets",
      emptyInbox: "No conversations yet", emptyTickets: "No open tickets",
      topics: "Sample questions — verified to be answered from the documents",
      topicsNone: "No verified sample questions right now",
      roomNew: "New conversation", roomNewSub: "Type the member's question as asked, or pick a sample question below.",
      roomThread: "Conversation", roomMsgs: "messages", roomSince: "started", roomTag: "Document-grounded replies · evidence verbatim",
      placeholder: "Type the member's question here…",
      hint: "Enter sends · Shift+Enter adds a line",
      send: "Send",
      settingsTitle: "Settings",
      settingsBody: "Secrets are never kept on this screen — they live only in the server's secret store and the provider dashboards. Below are the service's real capabilities and this desk's rules.",
      capsTitle: "Capabilities (/v1/capabilities)", rulesTitle: "Terminology and format rules",
      yes: "available", no: "not deployed",
      capRows: { answer_path: "Answer path", llm: "Language model", documents: "Documents", stt: "Speech → text (STT)", tts: "Text → speech (TTS)", grounding: "Refusal strings", rate_limit: "Rate limit", checked_at: "Checked" },
      llmUnreachable: "unreachable — the document passage is shown verbatim instead",
      mic: "Speak",
    },
  };

  const state = {
    lang: (function () { try { return localStorage.getItem("bhasa-lang") || "ne"; } catch (e) { return "ne"; } })(),
    view: "home",
    threads: [],
    tickets: [],
    caps: null,
    health: null,
    chatId: null,
    started: false,
    chips: { ne: [], en: [] },
    online: navigator.onLine,
    pending: null,
    fontOk: null,
  };

  const $ = (id) => document.getElementById(id);
  const t = () => CHROME[state.lang];
  const log = $("log");
  const box = $("box");
  const sendBtn = $("send");
  const form = $("form");
  const announceEl = $("announce");

  function announce(text) {
    if (!text) return;
    announceEl.textContent = "";
    setTimeout(() => { announceEl.textContent = text; }, 30);
  }

  function isDark() { return document.documentElement.classList.contains("dark"); }
  function setTheme(dark) {
    document.documentElement.classList.toggle("dark", dark);
    document.documentElement.style.colorScheme = dark ? "dark" : "light";
    try { localStorage.setItem("bhasa-theme", dark ? "dark" : "light"); } catch (e) {}
    paintTheme();
  }
  function paintTheme() {
    const c = t();
    const dark = isDark();
    const btn = $("theme-toggle");
    btn.setAttribute("aria-pressed", dark ? "true" : "false");
    btn.setAttribute("aria-label", c.themeToggle);
    $("theme-label").textContent = dark ? c.themeToLight : c.themeToDark;
    $("theme-color").setAttribute("content", dark ? "#0e1513" : "#f4fbf8");
  }

  /* ------------------------------------------------------------ language */

  function setLang(lang) {
    state.lang = lang === "en" ? "en" : "ne";
    try { localStorage.setItem("bhasa-lang", state.lang); } catch (e) {}
    paintChrome();
    renderInbox();
    renderCounts();
    renderChips();
    renderStatus();
    renderSettings();
    if (ime) ime.paintToggle();
    announce(state.lang === "ne" ? "इन्टरफेस नेपालीमा" : "Interface in English");
  }

  function paintLangGroup() {
    const c = t();
    $("lang-group").setAttribute("aria-label", c.langGroup);
    $("lang-group-label").textContent = c.langGroup;
    for (const code of ["ne", "en"]) {
      const btn = $("lang-" + code);
      const on = state.lang === code;
      btn.setAttribute("aria-checked", on ? "true" : "false");
      btn.classList.toggle("is-on", on);
      btn.tabIndex = on ? 0 : -1;
      // Non-colour indication: a check glyph and weight, plus aria-checked.
      btn.querySelector(".check").textContent = on ? "✓" : "";
    }
  }

  /* --------------------------------------------------------------- views */

  function setView(name) {
    state.view = name;
    const chat = name === "chat";
    $("stage").className = "stage " + (chat ? "chat-room" : name);
    $("app").classList.toggle("is-chat", chat);
    $("view-home").classList.toggle("hidden", name !== "home");
    $("view-chat").classList.toggle("hidden", !chat);
    $("chat-inbox").classList.toggle("hidden", !chat);
    $("view-settings").classList.toggle("hidden", name !== "settings");
    for (const v of ["home", "chat", "settings"]) {
      const b = $("nav-" + v);
      b.classList.toggle("active", name === v);
      b.setAttribute("aria-current", name === v ? "page" : "false");
    }
    paintChrome();
    if (chat) { if (!state.started) showEmpty(); box.focus(); }
    if (name === "settings") renderSettings();
  }

  function paintChrome() {
    const c = t();
    document.documentElement.lang = state.lang;
    document.title = c.title;
    $("brand-sub").textContent = c.brandSub;
    $("nav-home").textContent = c.home;
    $("nav-chat").textContent = c.chat;
    $("nav-settings").textContent = c.settings;
    $("page-title").textContent = { home: c.pageHome, chat: c.pageChat, settings: c.pageSettings }[state.view];
    $("home-title").textContent = c.homeTitle;
    $("home-lede").textContent = c.homeLede;
    $("stat-chat-label").textContent = c.statChat;
    $("stat-ticket-label").textContent = c.statTicket;
    $("stat-docs-label").textContent = c.statDocs;
    $("door-chat-kicker").textContent = c.doorChatKicker;
    $("door-chat-title").textContent = c.doorChatTitle;
    $("door-chat-body").textContent = c.doorChatBody;
    $("door-voice-kicker").textContent = c.doorVoiceKicker;
    $("door-voice-title").textContent = c.doorVoiceTitle;
    $("door-voice-body").textContent = c.doorVoiceBody(state.caps ? core.clock24(state.caps.checked_at, state.lang) : "—");
    $("chat-inbox-title").textContent = c.inbox;
    $("chat-new").textContent = c.newChat;
    $("tickets-label").textContent = c.tickets;
    $("topics-label").textContent = c.topics;
    $("room-tag").textContent = c.roomTag;
    $("hint").textContent = c.hint;
    $("cred-rule").textContent = ui.copy(state.lang).credRule;
    $("settings-title").textContent = c.settingsTitle;
    $("settings-body").textContent = c.settingsBody;
    $("caps-title").textContent = c.capsTitle;
    $("rules-title").textContent = c.rulesTitle;
    box.placeholder = c.placeholder;
    sendBtn.textContent = c.send;
    paintLangGroup();
    paintTheme();
    paintRoomHead();
    if (state.view === "chat" && !state.started) showEmpty();
  }

  /* --------------------------------------------------------- one store */

  async function loadThreads() {
    // The ONLY place conversations are fetched. Both counts render from state.threads.
    try {
      const [threads, tickets] = await Promise.all([
        fetch("/v1/support/conversations?channel=chat").then((r) => r.json()),
        fetch("/v1/support/tickets").then((r) => r.json()),
      ]);
      state.threads = Array.isArray(threads) ? threads : [];
      state.tickets = Array.isArray(tickets) ? tickets : [];
    } catch (e) {
      // keep the previous store; the health poll reports the outage
    }
    renderCounts();
    renderInbox();
    renderTickets();
  }

  function renderCounts() {
    const counts = core.deskCounts(state);
    $("stat-chat").textContent = core.fmtNum(counts.threads, state.lang);
    $("chat-inbox-count").textContent = core.fmtNum(counts.inbox, state.lang);
    $("stat-ticket").textContent = core.fmtNum(counts.openTickets, state.lang);
    $("stat-docs").textContent = core.fmtNum(state.caps ? state.caps.documents : 0, state.lang);
  }

  function renderInbox() {
    const c = t();
    const list = $("chat-list");
    list.innerHTML = "";
    if (!state.threads.length) {
      list.appendChild(ui.el("div", { class: "empty", text: c.emptyInbox }));
      return;
    }
    for (const row of state.threads) {
      const btn = ui.el("button", { type: "button", class: "convo" + (row.id === state.chatId ? " on" : ""), "aria-current": row.id === state.chatId ? "true" : "false" });
      btn.appendChild(ui.el("div", { class: "convo__top" }, [
        ui.el("span", { class: "convo__name", text: c.memberRow + " · " + row.id.slice(0, 8).toUpperCase() }),
        ui.el("time", { class: "convo__time", datetime: row.updated_at || "", text: core.relativeTime(row.updated_at, state.lang) }),
      ]));
      btn.appendChild(ui.el("div", { class: "convo__preview", text: row.preview || "—" }));
      // No raw locale code in the DOM (E17a); digits follow the interface language (E17b).
      btn.appendChild(ui.el("div", { class: "convo__meta", text: core.fmtNum(row.message_count || 0, state.lang) + " " + c.roomMsgs }));
      btn.addEventListener("click", () => openThread(row.id));
      list.appendChild(btn);
    }
  }

  function renderTickets() {
    const c = t();
    const host = $("ticket-list");
    host.innerHTML = "";
    const open = state.tickets.filter((r) => r.status !== "resolved");
    if (!open.length) { host.appendChild(ui.el("div", { class: "empty", text: c.emptyTickets })); return; }
    for (const row of open.slice(0, 8)) host.appendChild(ui.el("div", { class: "ticket", text: (row.id || "") + " · " + (row.subject || row.status) }));
  }

  /* ---------------------------------------------------------- thread */

  function paintRoomHead() {
    const c = t();
    if (!state.chatId) {
      $("room-title").textContent = c.roomNew;
      $("room-sub").textContent = c.roomNewSub;
      return;
    }
    const turns = log.querySelectorAll(".turn").length;
    const meta = state.threads.find((r) => r.id === state.chatId) || {};
    let sub = core.fmtNum(turns, state.lang) + " " + c.roomMsgs;
    if (meta.created_at) sub += " · " + c.roomSince + " " + core.relativeTime(meta.created_at, state.lang);
    $("room-title").textContent = c.roomThread + " " + state.chatId.slice(0, 8).toUpperCase();
    $("room-sub").textContent = sub;
  }

  function showEmpty() {
    log.innerHTML = "";
    log.appendChild(ui.emptyThread(state.lang));
    state.started = false;
    state.chatId = null;
    state.pending = null;
    paintRoomHead();
  }

  function append(node) {
    if (!state.started) { log.innerHTML = ""; state.started = true; }
    log.appendChild(node);
    log.scrollTop = log.scrollHeight;
    return node;
  }

  function renderReply(raw) {
    // The API reports the question's language as `language`; cards frame in that language (E10).
    const data = Object.assign({}, raw, { question_language: raw.language });
    if (data.kind === "refusal") {
      return ui.refusalCard(data, {
        onTicket: async (btn) => {
          btn.disabled = true;
          try {
            await fetch("/v1/support/tickets", {
              method: "POST", headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ subject: (state.pending && state.pending.text || "").slice(0, 80), description: data.reply, category: "other", conversation_id: state.chatId }),
            });
            btn.textContent = ui.copy(data.question_language).ticketOpened;
            announce(ui.copy(data.question_language).ticketOpened);
            loadThreads();
          } catch (e) { btn.disabled = false; }
        },
      });
    }
    if (data.kind === "credential_decline") return ui.credentialDeclineCard({ reply: data.reply, language: data.language });
    return ui.answerCard(data);
  }

  async function openThread(id) {
    let data;
    try { data = await fetch("/v1/support/conversations/" + id).then((r) => r.json()); } catch (e) { return; }
    state.chatId = id;
    state.started = true;
    state.pending = null;
    log.innerHTML = "";
    for (const msg of data.messages || []) {
      const meta = msg.meta || {};
      if (msg.role === "user") {
        log.appendChild(ui.memberTurn({ text: msg.content, lang: meta.language || core.detectLanguage(msg.content), redacted: !!meta.redacted }));
      } else if (meta.kind === "refusal") {
        log.appendChild(ui.refusalCard({ reply: msg.content, refusal_type: meta.refusal_type, question_language: meta.language }));
      } else if (meta.kind === "credential_decline") {
        log.appendChild(ui.credentialDeclineCard({ reply: msg.content, language: core.detectLanguage(msg.content) }));
      } else {
        log.appendChild(ui.answerCard({ reply: msg.content, passage: meta.passage || null, question_language: meta.language || core.detectLanguage(msg.content) }));
      }
    }
    log.scrollTop = log.scrollHeight;
    paintRoomHead();
    renderInbox();
    box.focus();
  }

  /* ------------------------------------------------------------- send */

  // A local 3.4 GB Gemma needs ~13 s of eval (more on first load), so the
  // "taking longer" line appears at 8 s and the hard timeout is 60 s.
  const TIMEOUT_MS = 60000;
  const LONG_MS = 8000;

  async function send(text) {
    const message = core.nfc(text).trim();
    if (!message || sendBtn.disabled) return;
    if (state.pending && state.pending.inflight) return;
    const shapes = core.credentialShapes(message);
    if (shapes.length) {
      showComposerNotice("credential");
      announce(ui.copy(state.lang).credWarn);
      return;
    }
    const qlang = core.detectLanguage(message);
    // One user action = one thread row. A retry reuses this pending object.
    const turn = append(ui.memberTurn({ text: message, lang: qlang, pending: true }));
    state.pending = { text: message, lang: qlang, turn, attempts: 0, inflight: false };
    await deliver();
  }

  async function deliver() {
    const p = state.pending;
    if (!p) return;
    p.attempts += 1;
    p.inflight = true;
    sendBtn.disabled = true;
    const c = ui.copy(p.lang);
    const status = append(ui.searching(p.lang));
    announce(c.searching);
    const ctrl = new AbortController();
    const longTimer = setTimeout(() => { ui.escalateSearching(status, p.lang, () => ctrl.abort("cancel")); announce(c.searchingLong); }, LONG_MS);
    const killTimer = setTimeout(() => ctrl.abort("timeout"), TIMEOUT_MS);
    let outcome = null;
    try {
      const res = await fetch("/v1/support/chat", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: p.text, conversation_id: state.chatId, locale: "auto", channel: "chat" }),
        signal: ctrl.signal,
      });
      if (res.status === 429) {
        const ra = res.headers.get("Retry-After");
        outcome = { kind: "rate-limit", retryAfter: ra && /^\d+$/.test(ra) ? Number(ra) : null };
      } else if (!res.ok) {
        outcome = { kind: "failure", reason: "http_" + res.status };
      } else {
        outcome = { kind: "ok", data: await res.json() };
      }
    } catch (err) {
      const reason = ctrl.signal.aborted ? (ctrl.signal.reason === "timeout" ? "timeout" : "cancel") : "network";
      outcome = { kind: "failure", reason };
    } finally {
      clearTimeout(longTimer); clearTimeout(killTimer);
      status.remove();
      p.inflight = false;
      sendBtn.disabled = !state.online;
    }
    if (outcome.kind === "ok") {
      const data = outcome.data;
      state.chatId = data.conversation_id;
      p.turn.setAttribute("data-state", "sent");
      if (data.kind === "credential_decline") {
        p.turn.querySelector(".bubble").textContent = data.echo; // never echo the secret
        p.turn.querySelector(".who").textContent += " · " + c.credDecline;
      }
      append(renderReply(data));
      announce(data.kind === "refusal" ? c.refusalLabel[data.refusal_type === "quantity" ? "quantity" : "general"] : data.kind === "credential_decline" ? c.credDecline : c.quoted);
      state.pending = null;
      paintRoomHead();
      loadThreads();
      return;
    }
    if (outcome.kind === "rate-limit") {
      const card = append(ui.rateLimitCard({ lang: p.lang, retryAfter: outcome.retryAfter, onRetry: () => { card.remove(); deliver(); } }));
      announce(c.rateLimited);
      return;
    }
    if (outcome.reason === "cancel") {
      const card = append(ui.failureCard({ lang: p.lang, reason: "cancel", onRetry: () => { card.remove(); deliver(); } }));
      announce(c.failed);
      return;
    }
    const card = append(ui.failureCard({ lang: p.lang, reason: outcome.reason === "timeout" ? "timeout" : "network", onRetry: () => { card.remove(); deliver(); } }));
    announce(outcome.reason === "timeout" ? c.timeout : c.failed);
    if (outcome.reason === "network") pollHealth();
  }

  /* ------------------------------------------------------- composer */

  function showComposerNotice(kind) {
    const slot = $("composer-notice");
    slot.innerHTML = "";
    if (!kind) return;
    slot.appendChild(ui.composerNotice({ kind, lang: state.lang }));
  }

  function checkComposer() {
    const shapes = core.credentialShapes(box.value);
    const current = $("composer-notice").firstChild;
    if (shapes.length) { if (!current || current.getAttribute("data-state") !== "credential") showComposerNotice("credential"); }
    else if (current && current.getAttribute("data-state") === "credential") showComposerNotice(state.fontOk === false ? "font" : null);
    sendBtn.setAttribute("aria-disabled", shapes.length ? "true" : "false");
  }

  function growBox() {
    box.style.height = "auto";
    box.style.height = Math.min(box.scrollHeight, 168) + "px";
  }

  function renderChips() {
    const c = t();
    const wrap = $("chips");
    wrap.innerHTML = "";
    const list = state.chips[state.lang] || [];
    if (!list.length) { wrap.appendChild(ui.el("span", { class: "chips__none", text: c.topicsNone })); return; }
    for (const q of list) {
      const btn = ui.el("button", { type: "button", class: "chip", text: q, lang: state.lang });
      btn.addEventListener("click", () => send(q));
      wrap.appendChild(btn);
    }
  }

  /* ------------------------------------------------ capabilities/health */

  async function loadCaps() {
    try { state.caps = await fetch("/v1/capabilities").then((r) => r.json()); } catch (e) { state.caps = null; }
    renderVoice();
    renderCounts();
    paintChrome();
    renderSettings();
    try {
      const topics = await fetch("/v1/support/topics").then((r) => r.json());
      state.chips = topics.chips || { ne: [], en: [] };
    } catch (e) { state.chips = { ne: [], en: [] }; }
    renderChips();
  }

  function renderVoice() {
    // Voice controls exist in the DOM ONLY when /capabilities says available === true.
    const stt = state.caps && state.caps.stt && state.caps.stt.available === true;
    const host = $("voice-slot");
    host.innerHTML = "";
    if (!stt) return;
    const btn = ui.el("button", { type: "button", id: "mic", class: "btn ghost", text: "🎤 " + t().mic, "aria-label": t().mic });
    btn.addEventListener("click", () => {
      const Rec = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!Rec) { showComposerNotice("unsupported"); return; }
      try {
        const rec = new Rec();
        rec.lang = state.lang === "ne" ? "ne-NP" : "en-US";
        rec.onresult = (ev) => { const txt = ev.results[0][0].transcript; if (txt) { box.value = txt; growBox(); box.focus(); } };
        rec.onerror = (ev) => showComposerNotice(ev.error === "not-allowed" ? "mic-blocked" : "mic-fail");
        rec.start();
      } catch (e) { showComposerNotice("mic-fail"); }
    });
    host.appendChild(btn);
  }

  async function pollHealth() {
    const c = t();
    try {
      const res = await fetch("/v1/health", { cache: "no-store" });
      const data = await res.json();
      state.health = { ok: !!data.ok && res.ok, checkedAt: new Date().toISOString(), server: data };
    } catch (e) {
      state.health = { ok: false, checkedAt: new Date().toISOString() };
    }
    renderStatus();
    renderNetBanner();
  }

  function renderStatus() {
    const c = t();
    const pill = $("live");
    const glyph = $("status-glyph");
    if (!state.health) { pill.textContent = c.statusChecking; glyph.textContent = "◌"; pill.parentNode.setAttribute("data-state", "checking"); return; }
    const when = core.clock24(state.health.checkedAt, state.lang);
    if (state.health.ok) { pill.textContent = c.statusOpen(when); glyph.textContent = "●"; pill.parentNode.setAttribute("data-state", "open"); }
    else { pill.textContent = c.statusDown(when); glyph.textContent = "✕"; pill.parentNode.setAttribute("data-state", "down"); }
  }

  function renderNetBanner() {
    const host = $("net-banner");
    host.innerHTML = "";
    if (!state.online) host.appendChild(ui.offlineBanner(state.lang, "offline"));
    else if (state.health && !state.health.ok) host.appendChild(ui.offlineBanner(state.lang, "service"));
    sendBtn.disabled = !state.online || (state.pending && state.pending.inflight);
  }

  function renderSettings() {
    const c = t();
    const host = $("caps-table");
    host.innerHTML = "";
    const caps = state.caps;
    if (!caps) { host.appendChild(ui.el("div", { class: "empty", text: c.statusChecking })); return; }
    const llm = caps.llm || {};
    const rows = [
      [c.capRows.answer_path, caps.answer_path],
      [c.capRows.llm, llm.reachable ? `${llm.backend} · ${llm.model}` : `${llm.backend || "—"} · ${c.llmUnreachable}`],
      [c.capRows.documents, core.fmtNum(caps.documents, state.lang)],
      [c.capRows.stt, caps.stt.available ? c.yes : c.no + " (" + caps.stt.reason + ")"],
      [c.capRows.tts, caps.tts.available ? c.yes : c.no + " (" + caps.tts.reason + ")"],
      [c.capRows.grounding, caps.grounding.available ? (caps.grounding.refusal_strings.source + " · " + core.clock24(caps.grounding.refusal_strings.fetched_at, state.lang)) : c.no],
      [c.capRows.rate_limit, caps.rate_limit.enabled ? c.yes : c.no],
      [c.capRows.checked_at, core.clock24(caps.checked_at, state.lang)],
    ];
    const dl = ui.el("dl", { class: "caps" });
    for (const [k, v] of rows) { dl.appendChild(ui.el("dt", { text: k })); dl.appendChild(ui.el("dd", { text: String(v) })); }
    host.appendChild(dl);
    fetch("/v1/terminology").then((r) => r.json()).then((terms) => {
      const ul = $("rules-list");
      ul.innerHTML = "";
      const items = [terms.product_name.rule, terms.numerals.rule, terms.time.rule, terms.calendar.rule, "Normalization: " + terms.normalization.form + " only, never " + terms.normalization.never + ".", terms.refusal_strings.rule];
      for (const it of items) ul.appendChild(ui.el("li", { text: it, lang: "en" }));
    }).catch(() => {});
  }

  /* ---------------------------------------------------------- font check
     E13: the browser's font availability check returns true for a fallback.
     Measure real glyph widths on a canvas after fonts settle. */
  function checkDevanagariFont() {
    try {
      const probe = "नमस्ते बैंक खाता ब्याजदर";
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");
      const w = (family) => { ctx.font = `20px ${family}`; return ctx.measureText(probe).width; };
      const target = w('"Noto Sans Devanagari", monospace');
      const mono = w("monospace");
      const target2 = w('"Noto Sans Devanagari", serif');
      const serif = w("serif");
      state.fontOk = !(Math.abs(target - mono) < 0.5 && Math.abs(target2 - serif) < 0.5);
      document.documentElement.setAttribute("data-devanagari-font", state.fontOk ? "loaded" : "fallback");
      if (!state.fontOk) { showComposerNotice("font"); announce(ui.copy(state.lang).fontWarn); }
    } catch (e) { state.fontOk = null; }
  }

  /* --------------------------------------------------------------- IME */

  const ime = window.DeskIME.attach(box, {
    strip: $("ime-strip"),
    toggle: $("ime-toggle"),
    lang: () => state.lang,
    announce,
    fetchRun: async (fullText, runStart, choices) => {
      // Whole composer as context; the server decides the run at `runStart`.
      const res = await fetch("/v1/support/translit/candidates", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: fullText, choices }),
      });
      const data = await res.json();
      return (data.runs || []).find((r) => r.kind === "latin" && r.start === runStart) || null;
    },
  });

  /* ------------------------------------------------------------- wiring */

  document.querySelectorAll(".sidebar nav button").forEach((btn) => btn.addEventListener("click", () => setView(btn.getAttribute("data-view"))));
  document.querySelectorAll(".door[data-go]").forEach((btn) => btn.addEventListener("click", () => setView(btn.getAttribute("data-go"))));
  $("lang-ne").addEventListener("click", () => setLang("ne"));
  $("lang-en").addEventListener("click", () => setLang("en"));
  $("lang-group").addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft" || e.key === "ArrowRight" || e.key === "ArrowUp" || e.key === "ArrowDown") {
      e.preventDefault(); const next = state.lang === "ne" ? "en" : "ne"; setLang(next); $("lang-" + next).focus();
    }
  });
  $("theme-toggle").addEventListener("click", () => setTheme(!isDark()));
  $("chat-new").addEventListener("click", () => { showEmpty(); renderInbox(); box.focus(); });
  box.addEventListener("input", () => { growBox(); checkComposer(); });
  box.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" || event.shiftKey || event.isComposing) return;
    if (ime.open) return; // handled by the IME (accept, not send)
    event.preventDefault();
    form.requestSubmit();
  });
  box.addEventListener("ime-submit", () => form.requestSubmit());
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    await ime.commitPending();
    const message = box.value.trim();
    if (core.credentialShapes(message).length) { showComposerNotice("credential"); announce(ui.copy(state.lang).credWarn); return; }
    box.value = "";
    growBox();
    send(message);
  });
  window.addEventListener("online", () => { state.online = true; renderNetBanner(); announce(ui.copy(state.lang).online); });
  window.addEventListener("offline", () => { state.online = false; renderNetBanner(); announce(ui.copy(state.lang).offline); });
  document.addEventListener("visibilitychange", () => { if (!document.hidden) pollHealth(); });

  (document.fonts ? document.fonts.ready : Promise.resolve()).then(checkDevanagariFont);
  setView("home");
  loadCaps();
  loadThreads();
  pollHealth();
  setInterval(pollHealth, 30000);
})();
