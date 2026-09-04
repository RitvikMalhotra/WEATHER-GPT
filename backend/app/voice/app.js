/* WeatherGPT client.
 *
 * A thin shell over the existing API. Speech recognition and synthesis run in
 * the browser; every weather question goes to /api/v1/ai/chat and every place
 * lookup to /api/v1/locations/search, exactly as any other client would.
 *
 * Two rules this file keeps:
 *   1. No weather value is authored here. Figures are copied from the typed
 *      backend response for the turn, or from the answer text the backend
 *      rendered. A failed request produces a message, never a number.
 *   2. Times and dates are reformatted for reading only — 12-hour clock and
 *      DD-MM-YYYY — and nothing is ever sent back to the backend reformatted.
 */
(() => {
  "use strict";

  const CHAT_URL = "/api/v1/ai/chat";
  const LOCATIONS_URL = "/api/v1/locations/search";
  const REVERSE_URL = "/api/v1/locations/reverse";
  const STORE_PLACE = "weathergpt.place";
  const STORE_LANG = "weathergpt.lang";
  const STORE_CLIENT = "weathergpt.client";

  /* Offered only where the whole path works: the backend renders answers from
     an English or Hindi catalog, and the browser recognises and speaks both.
     Adding a language is one row here plus its catalog in app/ai/grounding.py;
     listing one without the catalog would claim support that does not exist. */
  const LANGUAGES = [
    { tag: "en-IN", base: "en", label: "English", native: "English" },
    { tag: "hi-IN", base: "hi", label: "Hindi", native: "हिंदी" },
  ];

  /* Client-authored chrome only. Weather values and their labels always come
     from the backend, in whatever language it answered. */
  const UI = {
    en: {
      working: "WeatherGPT is checking the latest weather data…",
      workingPast: "Looking through past weather data…",
      workingFuture: "Checking the forecast ahead…",
      workingPlace: "Resolving the location…",
      verdict: "Verdict",
      observed: "Observed",
      useMyLocation: "Use my current location",
      locating: "Finding your location…",
      locationDenied: "Location permission is off, so search for a place instead.",
      locationUnavailable: "Your device couldn’t provide a location just now.",
      noHistory: "Historical weather data isn’t available for that time and location.",
      watchAdded: (place) => `Now watching ${place}.`,
      watchRemoved: "Stopped watching that location.",
      watchFailed: "WeatherGPT couldn’t save that watched location.",
      watchNeedsDb: "Watched locations need the WeatherGPT database, which isn’t reachable right now.",
      watchEmpty: "No locations watched yet. Add one above and WeatherGPT will keep an eye on it.",
      watchMonitoring: "Monitoring active",
      watchPaused: "Monitoring paused",
      watchPending: "Not checked yet",
      watchChecked: (when) => `Last checked ${when}`,
      watchNoAlerts: "No active weather alerts",
      watchAmbiguous: "Several places share that name — pick the one you meant.",
      unreachable: "WeatherGPT couldn’t reach the weather service right now.",
      unreachableHint: "Nothing is shown rather than guessed. Check that the API is running, then try again.",
      failed: "WeatherGPT couldn’t complete that request.",
      failedHint: "No weather values are shown, because none were returned.",
      retry: "Try again",
      source: "Source",
      fetched: "fetched",
      modelRun: "run",
      cached: "Cached",
      whichPlace: "Which location did you mean?",
      placeUsed: "used for this answer",
      listening: "Listening…",
      dictating: "Listening — your words will appear in the box.",
      thinking: "Fetching weather data…",
      answeredWith: (names) => `Answered using ${names} data.`,
      answered: "Answered.",
      needsDetail: "WeatherGPT needs one more detail.",
      langSet: (name) => `${name} selected.`,
      noVoiceInput: "Voice input isn’t available in this browser. Chrome or Edge supports it — typing works everywhere.",
      noVoiceOutput: "This browser can’t speak the answer, but it is shown above.",
      noVoiceFor: (name) => `No ${name} voice is installed on this device, so the answer is shown but not spoken.`,
      observed: "Observed",
      colTime: "Time",
      colDay: "Day",
      colConditions: "Conditions",
      showingFirst: (shown, total) => `Showing the first ${shown} of ${total} returned points.`,
      placeSet: (name) => `Location set to ${name}.`,
      searching: "Searching…",
      noPlaces: (q) => `We couldn’t find that location. Nothing matched “${q}”.`,
      placeSearchFailed: "Location search is unavailable right now.",
      typeMore: "Type at least two letters. Places that share a name are listed with their state, so you can pick the right one.",
    },
    hi: {
      working: "WeatherGPT ताज़ा मौसम डेटा देख रहा है…",
      workingPast: "पिछले मौसम के आंकड़े देखे जा रहे हैं…",
      workingFuture: "आगे का पूर्वानुमान देखा जा रहा है…",
      workingPlace: "स्थान पता किया जा रहा है…",
      verdict: "निष्कर्ष",
      observed: "प्रेक्षण",
      useMyLocation: "मेरा वर्तमान स्थान इस्तेमाल करें",
      locating: "आपका स्थान खोजा जा रहा है…",
      locationDenied: "स्थान की अनुमति बंद है, कृपया जगह खोजें।",
      locationUnavailable: "आपका डिवाइस अभी स्थान नहीं दे सका।",
      noHistory: "उस समय और स्थान के लिए पिछला मौसम डेटा उपलब्ध नहीं है।",
      watchAdded: (place) => `${place} पर नज़र रखी जा रही है।`,
      watchRemoved: "उस स्थान पर नज़र रखना बंद कर दिया गया।",
      watchFailed: "WeatherGPT उस स्थान को सहेज नहीं सका।",
      watchNeedsDb: "नज़र रखी गई जगहों के लिए WeatherGPT डेटाबेस चाहिए, जो अभी उपलब्ध नहीं है।",
      watchEmpty: "अभी कोई स्थान नहीं जोड़ा गया। ऊपर एक जोड़ें, WeatherGPT उस पर नज़र रखेगा।",
      watchMonitoring: "निगरानी चालू",
      watchPaused: "निगरानी रुकी हुई",
      watchPending: "अभी जाँच नहीं हुई",
      watchChecked: (when) => `अंतिम जाँच ${when}`,
      watchNoAlerts: "कोई सक्रिय मौसम चेतावनी नहीं",
      watchAmbiguous: "इस नाम की कई जगहें हैं — सही वाली चुनें।",
      unreachable: "WeatherGPT अभी मौसम सेवा तक नहीं पहुँच सका।",
      unreachableHint: "अनुमान लगाने के बजाय कुछ नहीं दिखाया गया है। जाँचें कि API चल रहा है, फिर दोबारा कोशिश करें।",
      failed: "WeatherGPT यह अनुरोध पूरा नहीं कर सका।",
      failedHint: "कोई मान नहीं मिला, इसलिए कोई मौसम मान नहीं दिखाया गया।",
      retry: "दोबारा कोशिश करें",
      source: "स्रोत",
      fetched: "प्राप्त",
      modelRun: "मॉडल रन",
      cached: "कैश से",
      whichPlace: "आपका मतलब किस जगह से था?",
      placeUsed: "इस उत्तर में उपयोग किया गया",
      listening: "सुन रहा हूँ…",
      dictating: "सुन रहा हूँ — आपके शब्द बॉक्स में आएँगे।",
      thinking: "मौसम डेटा लाया जा रहा है…",
      answeredWith: (names) => `${names} डेटा से उत्तर दिया गया।`,
      answered: "उत्तर दिया गया।",
      needsDetail: "WeatherGPT को एक और जानकारी चाहिए।",
      langSet: (name) => `${name} चुनी गई।`,
      noVoiceInput: "इस ब्राउज़र में वॉइस इनपुट उपलब्ध नहीं है। Chrome या Edge इसे समर्थन करते हैं — टाइप करना हर जगह काम करता है।",
      noVoiceOutput: "यह ब्राउज़र उत्तर बोल नहीं सकता, लेकिन वह ऊपर दिखाया गया है।",
      noVoiceFor: (name) => `इस डिवाइस पर ${name} आवाज़ स्थापित नहीं है, इसलिए उत्तर दिखाया गया है पर बोला नहीं गया।`,
      observed: "प्रेक्षण",
      colTime: "समय",
      colDay: "दिन",
      colConditions: "स्थिति",
      showingFirst: (shown, total) => `${total} में से पहले ${shown} बिंदु दिखाए गए हैं।`,
      placeSet: (name) => `स्थान ${name} सेट किया गया।`,
      searching: "खोजा जा रहा है…",
      noPlaces: (q) => `वह स्थान नहीं मिला। “${q}” से कुछ मेल नहीं खाया।`,
      placeSearchFailed: "स्थान खोज अभी उपलब्ध नहीं है।",
      typeMore: "कम से कम दो अक्षर लिखें। एक ही नाम वाली जगहें उनके राज्य के साथ दिखाई जाती हैं।",
    },
  };

  /* WMO condition category -> the icon that draws it and the sky it implies. */
  const CONDITIONS = {
    clear: ["ic-sun", "clear"],
    mainly_clear: ["ic-sun", "clear"],
    partly_cloudy: ["ic-cloud-sun", "clouds"],
    overcast: ["ic-cloud", "clouds"],
    fog: ["ic-fog", "fog"],
    drizzle: ["ic-drizzle", "rain"],
    freezing_drizzle: ["ic-drizzle", "rain"],
    rain: ["ic-rain", "rain"],
    freezing_rain: ["ic-rain", "rain"],
    snow: ["ic-snow", "snow"],
    rain_showers: ["ic-rain", "rain"],
    snow_showers: ["ic-snow", "snow"],
    thunderstorm: ["ic-storm", "storm"],
    thunderstorm_with_hail: ["ic-storm", "storm"],
    unknown: ["ic-unknown", null],
  };

  const $ = (id) => document.getElementById(id);
  const el = {
    thread: $("thread"), threadWrap: $("threadWrap"), opening: $("opening"),
    composer: $("composer"), prompt: $("prompt"), send: $("send"), dictate: $("dictate"),
    speak: $("speak"), replay: $("replay"), stop: $("stop"), status: $("status"),
    chips: $("chips"),
    placeButton: $("placeButton"), placeLabel: $("placeLabel"), placePanel: $("placePanel"),
    placeInput: $("placeInput"), placeResults: $("placeResults"), placeHint: $("placeHint"),
    useLocation: $("useLocation"),
    langButton: $("langButton"), langLabel: $("langLabel"), langMenu: $("langMenu"),
    alertsButton: $("alertsButton"), alertsPanel: $("alertsPanel"), alertsCount: $("alertsCount"),
    watchInput: $("watchInput"), watchResults: $("watchResults"), watchHint: $("watchHint"),
    watchList: $("watchList"), watchConfirm: $("watchConfirm"),
    watchConfirmPlace: $("watchConfirmPlace"), watchCreate: $("watchCreate"),
    watchCancel: $("watchCancel"),
  };

  const state = {
    language: LANGUAGES[0],
    place: null,          // the location this conversation is about
    sessionId: null,      // the backend's own follow-up context
    busy: false,
    lastSpoken: "",
    listening: false,
    micMode: "ask",
  };

  const t = () => UI[state.language.base] || UI.en;

  /* ---------------------------------------------------------------- storage */

  function remember(key, value) {
    try { window.localStorage.setItem(key, JSON.stringify(value)); } catch { /* private mode */ }
  }
  function recall(key) {
    try {
      const raw = window.localStorage.getItem(key);
      return raw ? JSON.parse(raw) : null;
    } catch { return null; }
  }

  /* ------------------------------------------------------- time and dates --
   * Presentation only. The backend's own timestamps are never rewritten on the
   * way out; these functions run on text and values coming back in.
   */

  const ISO_DATETIME = /\b(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)(Z|[+-]\d{2}:?\d{2})?/g;
  const ISO_DATE = /\b\d{4}-\d{2}-\d{2}\b/g;
  // Stateless twin of ISO_DATETIME, for testing a value without moving the
  //  global regex's cursor.
  const LOOKS_LIKE_STAMP = /\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}/;

  function zone() {
    return (state.place && state.place.timezone) || undefined;
  }

  /* Backend datetimes are UTC (app.domain declares it), so an ISO string with
     no offset is read as UTC rather than as the reader's wall clock. */
  function toDate(value) {
    if (value instanceof Date) return value;
    const text = String(value);
    const hasOffset = /(?:Z|[+-]\d{2}:?\d{2})$/.test(text);
    const date = new Date(/\d{4}-\d{2}-\d{2}T/.test(text) && !hasOffset ? `${text}Z` : text);
    return Number.isNaN(date.getTime()) ? null : date;
  }

  function formatTime(value) {
    const date = toDate(value);
    if (!date) return String(value);
    return new Intl.DateTimeFormat("en-IN", {
      hour: "numeric", minute: "2-digit", hour12: true, timeZone: zone(),
    }).format(date).toUpperCase().replace(/\s+/g, " ");
  }

  function formatDate(value) {
    const date = toDate(value);
    if (!date) return String(value);
    const parts = new Intl.DateTimeFormat("en-GB", {
      day: "2-digit", month: "2-digit", year: "numeric", timeZone: zone(),
    }).formatToParts(date);
    const get = (type) => (parts.find((p) => p.type === type) || {}).value || "";
    return `${get("day")}-${get("month")}-${get("year")}`;
  }

  function formatDateOnly(value) {
    // A bare calendar date has no instant, so it must not be shifted by a zone.
    const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(String(value));
    return match ? `${match[3]}-${match[2]}-${match[1]}` : formatDate(value);
  }

  function formatStamp(value) {
    const date = toDate(value);
    return date ? `${formatDate(date)}, ${formatTime(date)}` : String(value);
  }

  function formatWeekday(value) {
    const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(String(value));
    const date = match ? new Date(Date.UTC(+match[1], +match[2] - 1, +match[3])) : toDate(value);
    if (!date) return String(value);
    return new Intl.DateTimeFormat("en-IN", { weekday: "short", timeZone: "UTC" }).format(date);
  }

  /* Rewrites every ISO stamp the backend put inside rendered prose. Nothing
     else in the string is touched. */
  function humanize(text) {
    if (!text) return "";
    return text
      .replace(ISO_DATETIME, (whole) => formatStamp(whole))
      .replace(ISO_DATE, (whole) => formatDateOnly(whole));
  }

  /* --------------------------------------------------------------- numbers */

  // Mirrors Python's %g so a value read from typed JSON reads the same as the
  // one the backend already printed into the answer text.
  function g(value) {
    if (value === null || value === undefined || Number.isNaN(value)) return null;
    return String(Number(Number(value).toPrecision(6)));
  }

  /* ------------------------------------------------------------------- DOM */

  function make(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  function icon(id, className) {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    if (className) svg.setAttribute("class", className);
    svg.setAttribute("aria-hidden", "true");
    const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
    use.setAttribute("href", `#${id}`);
    svg.appendChild(use);
    return svg;
  }

  function say(message, tone) {
    el.status.dataset.tone = tone || "";
    el.status.textContent = message || "";
  }

  function scrollToLatest() {
    el.threadWrap.scrollTo({
      top: el.threadWrap.scrollHeight,
      behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth",
    });
  }

  function addTurn(side, node) {
    if (el.opening) { el.opening.remove(); el.opening = null; }
    const item = make("li", `turn turn--${side}`);
    item.appendChild(node);
    el.thread.appendChild(item);
    scrollToLatest();
    return item;
  }

  /* --------------------------------------------------------------- the API */

  async function postChat(message) {
    const response = await fetch(CHAT_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        session_id: state.sessionId,
        language: state.language.base,
      }),
    });
    if (!response.ok) {
      const error = new Error(`chat ${response.status}`);
      error.status = response.status;
      throw error;
    }
    return response.json();
  }

  async function searchLocations(query, limit, options) {
    const settings = options || {};
    let url = `${LOCATIONS_URL}?q=${encodeURIComponent(query)}&limit=${limit || 6}`;
    // The place the conversation is already on, so candidates near it rise.
    if (state.place && state.place.coordinates) {
      url += `&near_lat=${state.place.coordinates.latitude}`;
      url += `&near_lon=${state.place.coordinates.longitude}`;
    }
    const response = await fetch(url, { headers: { Accept: "application/json" } });
    if (!response.ok) throw new Error(`locations ${response.status}`);
    const body = await response.json();
    if (settings.full) return body;
    return Array.isArray(body.results) ? body.results : [];
  }

  async function nameThePoint(latitude, longitude) {
    const url = `${REVERSE_URL}?latitude=${latitude}&longitude=${longitude}`;
    const response = await fetch(url, { headers: { Accept: "application/json" } });
    if (!response.ok) throw new Error(`reverse ${response.status}`);
    return response.json();
  }

  /* ---------------------------------------------------------------- places */

  function displayName(location) {
    if (!location) return "";
    const parts = [location.name, location.admin1, location.country].filter(Boolean);
    if (parts.length) return parts.join(", ");
    const c = location.coordinates || {};
    return `${g(c.latitude)}, ${g(c.longitude)}`;
  }

  function shortName(location) {
    if (!location) return "";
    const parts = [location.name, location.admin1].filter(Boolean);
    return parts.length ? parts.join(", ") : displayName(location);
  }

  function setPlace(location, options) {
    if (!location || !location.coordinates) return;
    state.place = location;
    el.placeLabel.textContent = shortName(location);
    el.placeButton.title = displayName(location);
    // A bare coordinate pair is a position, not a place a person recognises on
    // their next visit, so only a named location is carried across sessions.
    if (location.name) remember(STORE_PLACE, location);
    renderChips();
    el.placeButton.dataset.set = "true";
    if (options && options.announce) say(t().placeSet(displayName(location)), "ok");
  }

  // "Miyāpur" and "Miyapur" are the same name written twice; a gazetteer may
  // return either spelling, and a mismatch here would hide the ambiguity.
  function foldName(value) {
    return String(value || "")
      .normalize("NFD")
      .replace(/[̀-ͯ]/g, "")
      .trim()
      .toLowerCase();
  }

  const near = (a, b) =>
    Math.abs(a.latitude - b.latitude) < 0.05 && Math.abs(a.longitude - b.longitude) < 0.05;

  /* The generic answer to "which Miyapur?": ask the gazetteer how many places
     carry the name that was actually resolved. Two or more in different
     administrative areas means the choice was not ours to make silently. There
     is no place list in this file, and no name is special-cased. */
  async function ambiguousAlternatives(location) {
    if (!location || !location.name) return null;
    // The gazetteer already decides whether a name is ambiguous, and it has
    // the administrative hierarchy to decide it with. Asking here a second
    // time, with a weaker rule, would disagree with it sooner or later.
    const asked = location.name.split(",")[0].trim();
    let body;
    try {
      body = await searchLocations(asked, 6, { full: true });
    } catch { return null; }
    if (!body.ambiguous) return null;

    const wanted = foldName(asked);
    const seen = new Set();
    const distinct = [];
    for (const candidate of body.results || []) {
      if (!candidate.name || foldName(candidate.name.split(",")[0]) !== wanted) continue;
      const key = `${candidate.admin1 || ""}|${candidate.country || ""}`.toLowerCase();
      if (seen.has(key)) continue;
      seen.add(key);
      distinct.push(candidate);
    }
    return distinct.length >= 2 ? distinct.slice(0, 4) : null;
  }

  /* -------------------------------------------------------------- rendering */

  function agentShell() {
    const wrapper = make("div", "agent");
    const head = make("div", "agent__head");
    const mark = make("span", "agent__mark");
    mark.setAttribute("aria-hidden", "true");
    mark.appendChild(icon("ic-mark"));
    head.appendChild(mark);
    head.appendChild(make("span", null, "WeatherGPT"));
    wrapper.appendChild(head);
    const card = make("div", "card");
    wrapper.appendChild(card);
    return { wrapper, card };
  }

  function renderUser(text) {
    return addTurn("user", make("div", "bubble--user", text));
  }

  /* Which kind of lookup is under way. A cosmetic reading of the words the
     person used — the backend decides what the question actually is, and this
     only chooses which sentence to show while waiting for it. */
  function waitingLine(question) {
    const text = String(question || "").toLowerCase();
    if (/\bago\b|yesterday|last night|\blast \w+|earlier|pehle|pahle|हुई|थी|पहले/.test(text)) {
      return t().workingPast;
    }
    if (/tomorrow|next|will |expected|forecast|hogi|hoga|कल|होगी|अगले|baad/.test(text)) {
      return t().workingFuture;
    }
    return t().working;
  }

  function renderWorking(question) {
    const { wrapper, card } = agentShell();
    const body = make("div", "card__body");
    const working = make("div", "working");
    const bars = make("span", "working__bars");
    bars.setAttribute("aria-hidden", "true");
    for (let i = 0; i < 4; i += 1) bars.appendChild(make("i"));
    working.appendChild(bars);
    working.appendChild(make("span", null, waitingLine(question)));
    body.appendChild(working);
    const skeleton = make("div", "skeleton");
    skeleton.setAttribute("aria-hidden", "true");
    for (let i = 0; i < 3; i += 1) skeleton.appendChild(make("i"));
    body.appendChild(skeleton);
    card.appendChild(body);
    return addTurn("agent", wrapper);
  }

  function renderError(kind, retryText) {
    const { wrapper, card } = agentShell();
    card.classList.add("card--error");
    const body = make("div", "card__body");
    body.appendChild(icon(kind === "offline" ? "ic-offline" : "ic-alert", "error__icon"));

    const column = make("div");
    const offline = kind === "offline";
    column.appendChild(make("p", "error__text", offline ? t().unreachable : t().failed));
    column.appendChild(make("p", "error__hint", offline ? t().unreachableHint : t().failedHint));
    if (retryText) {
      const actions = make("div", "error__actions");
      const button = make("button", "btn", t().retry);
      button.type = "button";
      button.addEventListener("click", () => { wrapper.closest("li").remove(); ask(retryText); });
      actions.appendChild(button);
      column.appendChild(actions);
    }
    body.appendChild(column);
    card.appendChild(body);
    return addTurn("agent", wrapper);
  }

  /* Splits a rendered answer into its opening sentence and its "- label: value"
     lines. The backend writes both; nothing is added here. */
  function splitAnswer(text) {
    const lead = [];
    const bullets = [];
    const tail = [];
    for (const raw of String(text || "").split("\n")) {
      const line = raw.trim();
      if (!line) continue;
      if (line.startsWith("- ")) bullets.push(line.slice(2));
      else if (bullets.length) tail.push(line);
      else lead.push(line);
    }
    return { lead: lead.join(" "), bullets, tail: tail.join("\n") };
  }

  function bulletParts(bullet) {
    const cut = bullet.indexOf(":");
    if (cut === -1) return { label: null, value: bullet.replace(/\.\s*$/, "") };
    return {
      label: bullet.slice(0, cut).trim(),
      value: bullet.slice(cut + 1).trim().replace(/\.\s*$/, ""),
    };
  }

  // "29 °C" -> ["29", "°C"]; a value with no unit keeps an empty second half.
  function splitUnit(value) {
    const match = /^(-?[\d.,]+)\s*(.*)$/.exec(value);
    return match ? [match[1], match[2]] : [value, ""];
  }

  /* A wind speed the backend prints as 1.4167 m/s is one decimal of real
     information and three of arithmetic. Shortening it is a reading choice, so
     the full figure stays available on the element rather than being lost. */
  function readable(number) {
    const value = Number(number);
    if (!Number.isFinite(value)) return { text: number, exact: null };
    const rounded = Math.round(value * 10) / 10;
    const text = String(rounded);
    return { text, exact: text === String(number) ? null : String(number) };
  }

  function conditionOf(code) {
    return CONDITIONS[code] || CONDITIONS.unknown;
  }

  /* When a question is answered from coordinates the backend has no name for
     the point and labels it "17.3989,78.4571". This puts back the name the
     person already chose for exactly those coordinates — it renames nothing
     and resolves nothing, it reuses a label already on screen. */
  function withKnownPlace(text, location) {
    if (!text || !location || location.name || !state.place) return text;
    const point = location.coordinates || {};
    if (!near(state.place.coordinates, point)) return text;
    const key = `${Number(point.latitude).toFixed(4)},${Number(point.longitude).toFixed(4)}`;
    return text.split(key).join(displayName(state.place));
  }

  function renderCurrent(result, card) {
    const data = result.data || {};
    const current = data.current || {};
    const body = make("div", "card__body");
    const { lead, bullets } = splitAnswer(result.answerText);

    if (lead) body.appendChild(make("p", "lede", humanize(withKnownPlace(lead, data.location))));

    const [iconId, sky] = conditionOf(current.condition);
    const night = current.is_day === false;
    const reading = make("div", "reading");
    reading.appendChild(icon(night && (sky === "clear") ? "ic-moon" : iconId, "reading__icon"));

    if (current.temperature_c !== null && current.temperature_c !== undefined) {
      const value = make("p", "reading__value", g(current.temperature_c));
      value.appendChild(make("span", "reading__unit", " °C"));
      reading.appendChild(value);
    }

    const descriptions = make("div");
    const description = current.condition_description
      || (current.condition && current.condition !== "unknown" ? current.condition.replace(/_/g, " ") : "");
    if (description) descriptions.appendChild(make("p", "reading__desc", description));
    if (current.observed_at) {
      descriptions.appendChild(make("p", "reading__sub", `${t().observed} ${formatStamp(current.observed_at)}`));
    }
    reading.appendChild(descriptions);
    body.appendChild(reading);

    /* The measured values come from the lines the backend already wrote, so
       their labels and units stay in the language it answered in. The three
       facts already shown above are not repeated. */
    const shownTemp = current.temperature_c !== null && current.temperature_c !== undefined
      ? `${g(current.temperature_c)} °C` : null;
    const metrics = make("div", "metrics");
    let index = 0;
    for (const bullet of bullets) {
      const { label, value } = bulletParts(bullet);
      if (!label || !value) continue;
      if (value === description || value === shownTemp) continue;
      if (LOOKS_LIKE_STAMP.test(value)) continue;
      const metric = make("div", "metric");
      metric.style.setProperty("--i", String(index));
      const labelNode = make("p", "metric__label", label);
      labelNode.title = label;
      metric.appendChild(labelNode);
      const [number, unit] = splitUnit(value);
      const shown = readable(number);
      const figure = make("p", "metric__value", shown.text);
      if (shown.exact) figure.title = `${shown.exact}${unit ? ` ${unit}` : ""}`;
      if (unit) figure.appendChild(make("span", "metric__unit", unit));
      metric.appendChild(figure);
      metrics.appendChild(metric);
      index += 1;
    }
    if (index) body.appendChild(metrics);

    card.appendChild(body);
    return night && (sky === "clear" || sky === "clouds") ? "night" : sky;
  }

  function figure(label, value, wet) {
    const span = make("span", `row__fig${wet ? " row__fig--wet" : ""}`);
    span.appendChild(make("span", null, `${label} `));
    span.appendChild(make("b", null, value));
    return span;
  }

  function renderForecast(result, card, hourly) {
    const data = result.data || {};
    const points = (hourly ? data.hourly : data.daily) || [];
    const body = make("div", "card__body");
    const { lead } = splitAnswer(result.answerText);
    if (lead) body.appendChild(make("p", "lede", humanize(withKnownPlace(lead, data.location))));

    if (!points.length) {
      body.appendChild(make("p", "prose", humanize(result.answerText)));
      card.appendChild(body);
      return null;
    }

    // An hour-by-hour answer is read for its shape, not row by row: eight
    // rows show the trend and leave the verdict — the part a person came
    // for — on screen with them.
    const limit = hourly ? 8 : 10;
    const shown = points.slice(0, limit);
    const rows = make("div", "rows");
    rows.setAttribute("role", "list");

    let sky = null;
    for (const point of shown) {
      const row = make("div", "row");
      row.setAttribute("role", "listitem");
      const when = hourly
        ? formatTime(point.valid_at)
        : `${formatWeekday(point.date)} ${formatDateOnly(point.date)}`;
      row.appendChild(make("span", "row__when", when));

      const [iconId, pointSky] = conditionOf(point.condition);
      if (!sky && pointSky) sky = pointSky;
      row.appendChild(icon(iconId, "row__icon"));
      row.appendChild(make("span", "row__what", point.condition_description || ""));

      const figures = make("span", "row__figures");
      if (hourly) {
        if (point.temperature_c != null) figures.appendChild(figure("", `${readable(g(point.temperature_c)).text}°`, false));
        if (point.precipitation_probability_pct != null) {
          figures.appendChild(figure("rain", `${g(point.precipitation_probability_pct)}%`, true));
        }
        if (point.wind_speed_ms != null) figures.appendChild(figure("wind", `${readable(g(point.wind_speed_ms)).text} m/s`, false));
      } else {
        if (point.temperature_min_c != null) figures.appendChild(figure("min", `${readable(g(point.temperature_min_c)).text}°`, false));
        if (point.temperature_max_c != null) figures.appendChild(figure("max", `${readable(g(point.temperature_max_c)).text}°`, false));
        if (point.precipitation_probability_max_pct != null) {
          figures.appendChild(figure("rain", `${g(point.precipitation_probability_max_pct)}%`, true));
        }
      }
      row.appendChild(figures);
      rows.appendChild(row);
    }
    body.appendChild(rows);
    if (points.length > shown.length) {
      body.appendChild(make("p", "more", t().showingFirst(shown.length, points.length)));
    }
    card.appendChild(body);
    return sky;
  }

  function renderCandidates(result, card, originalMessage) {
    const data = result.data || {};
    const results = Array.isArray(data.results) ? data.results : [];
    const body = make("div", "card__body");
    const { lead } = splitAnswer(result.answerText);
    body.appendChild(make("p", "lede", lead || t().whichPlace));
    if (results.length) {
      const list = make("div", "disambig__list");
      for (const candidate of results.slice(0, 6)) list.appendChild(candidateButton(candidate, originalMessage));
      body.appendChild(list);
    }
    card.appendChild(body);
    return null;
  }

  function candidateButton(candidate, question, current) {
    const button = make("button", "candidate");
    button.type = "button";
    if (current) button.setAttribute("aria-current", "true");
    button.appendChild(icon("ic-pin"));
    button.appendChild(make("span", null, displayName(candidate)));
    if (current) button.appendChild(make("span", "sr-only", ` — ${t().placeUsed}`));
    button.addEventListener("click", () => {
      setPlace(candidate, { announce: true });
      if (question) ask(question, { coordinates: candidate.coordinates, resend: true });
    });
    return button;
  }

  function renderHistory(result, card) {
    const data = result.data || {};
    const observations = Array.isArray(data.observations) ? data.observations : [];
    const body = make("div", "card__body");
    const { lead } = splitAnswer(result.answerText);
    if (lead) body.appendChild(make("p", "lede", humanize(withKnownPlace(lead, data.location))));

    if (!observations.length) {
      body.appendChild(make("p", "prose", t().noHistory));
      card.appendChild(body);
      return null;
    }

    const shown = observations.slice(0, 12);
    const rows = make("div", "rows");
    rows.setAttribute("role", "list");
    let sky = null;
    for (const entry of shown) {
      const weather = entry.weather || {};
      const row = make("div", "row");
      row.setAttribute("role", "listitem");
      row.appendChild(make("span", "row__when", formatTime(weather.observed_at)));

      const [iconId, pointSky] = conditionOf(weather.condition);
      if (!sky && pointSky) sky = pointSky;
      row.appendChild(icon(iconId, "row__icon"));
      row.appendChild(make("span", "row__what", weather.condition_description || ""));

      const figures = make("span", "row__figures");
      if (weather.temperature_c != null) {
        figures.appendChild(figure("", `${readable(g(weather.temperature_c)).text}°`, false));
      }
      if (weather.precipitation_mm != null) {
        figures.appendChild(figure("rain", `${readable(g(weather.precipitation_mm)).text} mm`, true));
      }
      if (weather.wind_speed_ms != null) {
        figures.appendChild(figure("wind", `${readable(g(weather.wind_speed_ms)).text} m/s`, false));
      }
      row.appendChild(figures);
      rows.appendChild(row);
    }
    body.appendChild(rows);
    if (observations.length > shown.length) {
      body.appendChild(make("p", "more", t().showingFirst(shown.length, observations.length)));
    }
    card.appendChild(body);
    return sky;
  }

  function renderProse(result, card) {
    const body = make("div", "card__body");
    body.appendChild(make("p", "prose", humanize(result.answerText)));
    card.appendChild(body);
    return null;
  }

  /* The one-line reading the backend computed from the values in this answer.
     It is rendered, never composed here: the client has no thresholds and no
     opinion of its own about what a number means. */
  function renderVerdict(card, verdict) {
    if (!verdict || !verdict.text) return;
    const block = make("div", "verdict");
    if (verdict.icon) {
      const glyph = make("span", "verdict__icon", verdict.icon);
      glyph.setAttribute("aria-hidden", "true");
      block.appendChild(glyph);
    }
    const column = make("div");
    const line = make("p", "verdict__text");
    line.appendChild(make("b", "verdict__label", `${t().verdict}: `));
    line.appendChild(document.createTextNode(verdict.text));
    column.appendChild(line);
    if (verdict.caveat) column.appendChild(make("p", "verdict__caveat", verdict.caveat));
    block.appendChild(column);
    card.appendChild(block);
  }

  function renderSource(card, sources) {
    if (!sources || !sources.length) return;
    const footer = make("div", "source");
    footer.appendChild(make("span", "source__label", t().source));

    sources.slice(0, 3).forEach((source, position) => {
      if (position) {
        const dot = make("span", "source__dot");
        dot.setAttribute("aria-hidden", "true");
        footer.appendChild(dot);
      }
      const group = make("span");
      group.appendChild(make("span", "source__name", source.provider_name || ""));
      // The source, and the model run behind it when there is a real one.
      // When it was fetched is plumbing, not provenance a reader needs.
      if (source.model_run_at) {
        group.appendChild(
          make("span", "source__meta", ` · ${t().modelRun} ${formatStamp(source.model_run_at)}`)
        );
      }
      footer.appendChild(group);
      if (source.cached) footer.appendChild(make("span", "source__tag", t().cached));
    });
    card.appendChild(footer);
  }

  function renderNote(card, text) {
    if (!text) return;
    const note = make("div", "note");
    note.appendChild(icon("ic-alert"));
    note.appendChild(make("p", null, text));
    card.appendChild(note);
  }

  async function renderAnswer(body, question) {
    const { wrapper, card } = agentShell();
    const result = (body.tool_results && body.tool_results[0]) || null;
    const payload = result ? { ...result, answerText: body.answer } : null;

    let sky = null;
    if (!payload) {
      const shell = make("div", "card__body");
      shell.appendChild(make("p", body.needs_clarification ? "lede" : "prose", humanize(body.answer)));
      card.appendChild(shell);
    } else if (payload.tool === "current_weather") {
      sky = renderCurrent(payload, card);
    } else if (payload.tool === "hourly_forecast" || payload.tool === "daily_forecast") {
      sky = renderForecast(payload, card, payload.tool === "hourly_forecast");
    } else if (payload.tool === "historical_weather") {
      sky = renderHistory(payload, card);
    } else if (payload.tool === "location_search") {
      renderCandidates(payload, card, question);
    } else {
      renderProse(payload, card);
    }

    renderVerdict(card, body.verdict);
    renderNote(card, body.safety_note);
    renderSource(card, body.sources);

    const turn = addTurn("agent", wrapper);
    if (sky) document.documentElement.dataset.sky = sky;

    /* Ambiguity is resolved in the column, after the answer, so a correct
       answer is never delayed by a check the user may not need. */
    const location = result && result.data && result.data.location;
    if (location && location.coordinates) {
      const alternatives = await ambiguousAlternatives(location);
      if (alternatives) {
        const block = make("div", "disambig");
        block.appendChild(make("p", "disambig__title", t().whichPlace));
        const list = make("div", "disambig__list");
        for (const candidate of alternatives) {
          list.appendChild(candidateButton(candidate, question, near(candidate.coordinates, location.coordinates)));
        }
        block.appendChild(list);
        card.insertBefore(block, card.querySelector(".source") || null);
        scrollToLatest();
      }
    }
    return turn;
  }

  /* ------------------------------------------------------------ asking ---- */

  function setBusy(busy) {
    state.busy = busy;
    el.prompt.disabled = busy;
    el.dictate.disabled = busy || !SpeechRecognitionAPI;
    el.speak.disabled = busy;
    updateSend();
    for (const chip of el.chips.querySelectorAll("button")) chip.disabled = busy;
  }

  async function ask(question, options) {
    const settings = options || {};
    const text = String(question || "").trim();
    if (!text || state.busy) return;

    if (!settings.silentUser) renderUser(text);
    setBusy(true);
    say(t().thinking);

    const working = renderWorking(text);
    /* A pinned location travels as coordinates appended to the message the
       backend parses; the person's own words are what is shown above. */
    const message = settings.coordinates
      ? `${text} at ${settings.coordinates.latitude}, ${settings.coordinates.longitude}`
      : text;

    let body;
    try {
      body = await postChat(message);
    } catch (error) {
      working.remove();
      setBusy(false);
      const offline = !error.status;
      say(offline ? t().unreachable : t().failed, "error");
      renderError(offline ? "offline" : "failed", text);
      return;
    }

    state.sessionId = body.session_id || state.sessionId;

    /* The backend asked for a location and this conversation already has one:
       supply it once rather than making the person repeat themselves. The
       backend then holds it for the rest of the session. */
    const canRetry = body.needs_clarification
      && body.intent !== "unknown"
      && body.intent !== "location_search"
      && !settings.coordinates
      && !settings.retried
      && state.place;
    if (canRetry) {
      working.remove();
      setBusy(false);
      return ask(text, {
        coordinates: state.place.coordinates,
        retried: true,
        silentUser: true,
      });
    }

    working.remove();
    await renderAnswer(body, text);

    const answered = result0Location(body);
    if (answered) setPlace(answered);

    const providers = [...new Set((body.sources || []).map((s) => s.provider_name).filter(Boolean))];
    if (body.needs_clarification) say(t().needsDetail, "warn");
    else if (providers.length) say(t().answeredWith(providers.join(" & ")), "ok");
    else say(t().answered, "ok");

    setBusy(false);
    speak(humanize(body.answer), body.language || state.language.base);
  }

  function result0Location(body) {
    const result = (body.tool_results && body.tool_results[0]) || null;
    const location = result && result.data && result.data.location;
    if (!location || !location.coordinates) return null;
    // An answer looked up by coordinates comes back unnamed; keeping the named
    // place the person picked is more useful than replacing it with a point.
    if (!location.name && state.place && near(state.place.coordinates, location.coordinates)) {
      return null;
    }
    return location;
  }

  /* ------------------------------------------------------------- composer */

  function updateSend() {
    el.send.disabled = state.busy || !el.prompt.value.trim();
  }

  function autoGrow() {
    el.prompt.style.height = "auto";
    el.prompt.style.height = `${Math.min(el.prompt.scrollHeight, 132)}px`;
  }

  /* The full invitation wraps to two lines on a phone, which makes an empty
     field look like a filled one. The short form says the same thing. */
  const NARROW = window.matchMedia("(max-width: 560px)");
  function fitPlaceholder() {
    el.prompt.placeholder = NARROW.matches
      ? "Ask about the weather…"
      : "Ask WeatherGPT anything about the weather…";
    autoGrow();
  }
  NARROW.addEventListener("change", fitPlaceholder);

  el.prompt.addEventListener("input", () => { autoGrow(); updateSend(); });
  el.prompt.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      el.composer.requestSubmit();
    }
  });

  el.composer.addEventListener("submit", (event) => {
    event.preventDefault();
    const text = el.prompt.value.trim();
    if (!text) return;
    el.prompt.value = "";
    autoGrow();
    updateSend();
    ask(text);
  });

  function renderChips() {
    const where = state.place && state.place.name ? ` in ${state.place.name}` : "";
    const questions = [
      `What is the weather${where}?`,
      `Is there rain expected in 48 hours${where}?`,
      `What was the rainfall yesterday${where}?`,
      `What will the wind speed be in 6 hours${where}?`,
      `7 day forecast${where}`,
    ];
    el.chips.textContent = "";
    for (const question of questions) {
      const chip = make("button", "chip", question);
      chip.type = "button";
      chip.disabled = state.busy;
      chip.addEventListener("click", () => ask(question));
      el.chips.appendChild(chip);
    }
  }

  /* ------------------------------------------------------------- popovers */

  const openPopovers = new Set();

  function closePopover(button, panel) {
    panel.hidden = true;
    button.setAttribute("aria-expanded", "false");
    openPopovers.delete(panel);
  }

  function openPopover(button, panel) {
    for (const other of [...openPopovers]) {
      other.hidden = true;
      const owner = document.querySelector(`[aria-controls="${other.id}"]`);
      if (owner) owner.setAttribute("aria-expanded", "false");
      openPopovers.delete(other);
    }
    panel.hidden = false;
    button.setAttribute("aria-expanded", "true");
    openPopovers.add(panel);
  }

  function togglePopover(button, panel) {
    if (panel.hidden) openPopover(button, panel);
    else closePopover(button, panel);
  }

  document.addEventListener("pointerdown", (event) => {
    for (const panel of [...openPopovers]) {
      const owner = document.querySelector(`[aria-controls="${panel.id}"]`);
      if (panel.contains(event.target) || (owner && owner.contains(event.target))) continue;
      closePopover(owner, panel);
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape" || !openPopovers.size) return;
    for (const panel of [...openPopovers]) {
      const owner = document.querySelector(`[aria-controls="${panel.id}"]`);
      closePopover(owner, panel);
      if (owner) owner.focus();
    }
  });

  /* ------------------------------------------------------ language menu -- */

  /* Names that mark a device's better engines. Matching on them is a
     preference, not a requirement: whatever the device actually has still gets
     used, and a language with no voice at all is reported rather than faked. */
  const PREFERRED_VOICE = /(natural|neural|online|google|premium|enhanced|siri)/i;

  function voiceFor(tag) {
    const voices = window.speechSynthesis ? window.speechSynthesis.getVoices() : [];
    if (!voices.length) return null;
    const wanted = tag.toLowerCase();
    const base = wanted.split("-")[0];
    const normalise = (v) => v.lang.toLowerCase().replace("_", "-");

    // An exact regional match first — en-IN reads an Indian place name the way
    // a person here says it, and en-US does not.
    const exact = voices.filter((v) => normalise(v) === wanted);
    const sameLanguage = voices.filter((v) => normalise(v).startsWith(base));
    for (const pool of [exact, sameLanguage]) {
      if (!pool.length) continue;
      return pool.find((v) => PREFERRED_VOICE.test(v.name)) || pool[0];
    }
    return null;
  }

  function renderLanguageMenu() {
    el.langMenu.textContent = "";
    for (const language of LANGUAGES) {
      const option = make("button", "menu__option");
      option.type = "button";
      option.setAttribute("role", "option");
      option.setAttribute("aria-selected", String(language.tag === state.language.tag));
      option.dataset.tag = language.tag;
      option.appendChild(icon("ic-check", "menu__check"));

      const label = make("span", "menu__label");
      label.appendChild(document.createTextNode(language.native));
      if (window.speechSynthesis && !voiceFor(language.tag)) {
        label.appendChild(make("span", "menu__note", "Text only on this device — no installed voice"));
      }
      option.appendChild(label);
      option.addEventListener("click", () => {
        selectLanguage(language.tag);
        closePopover(el.langButton, el.langMenu);
        el.langButton.focus();
      });
      el.langMenu.appendChild(option);
    }
    el.langMenu.appendChild(make("p", "menu__footnote",
      "WeatherGPT answers in the languages its data catalogue covers. More Indian languages appear here as they are added."));
  }

  function selectLanguage(tag) {
    const language = LANGUAGES.find((item) => item.tag === tag) || LANGUAGES[0];
    state.language = language;
    el.langLabel.textContent = language.native;
    // A new language is a new conversation: carried-over context is in the old
    // one and would answer the next question in the wrong register.
    state.sessionId = null;
    remember(STORE_LANG, language.tag);
    renderLanguageMenu();
    say(t().langSet(language.native), "ok");
  }

  el.langButton.addEventListener("click", () => togglePopover(el.langButton, el.langMenu));
  el.langMenu.addEventListener("keydown", (event) => {
    const options = [...el.langMenu.querySelectorAll(".menu__option")];
    const current = options.indexOf(document.activeElement);
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      const next = event.key === "ArrowDown"
        ? (current + 1) % options.length
        : (current - 1 + options.length) % options.length;
      options[next].focus();
    }
  });
  el.langButton.addEventListener("keydown", (event) => {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      openPopover(el.langButton, el.langMenu);
      const first = el.langMenu.querySelector(".menu__option");
      if (first) first.focus();
    }
  });

  /* ------------------------------------------------------ location picker */

  let searchTimer = null;
  let searchToken = 0;

  el.placeButton.addEventListener("click", () => {
    togglePopover(el.placeButton, el.placePanel);
    if (!el.placePanel.hidden) {
      el.placeInput.select();
      el.placeInput.focus();
    }
  });

  el.placeInput.addEventListener("input", () => {
    window.clearTimeout(searchTimer);
    const query = el.placeInput.value.trim();
    if (query.length < 2) {
      el.placeResults.textContent = "";
      el.placeHint.textContent = t().typeMore;
      el.placeHint.hidden = false;
      el.placeInput.setAttribute("aria-expanded", "false");
      return;
    }
    el.placeHint.textContent = t().searching;
    el.placeHint.hidden = false;
    searchTimer = window.setTimeout(() => runPlaceSearch(query), 280);
  });

  el.placeInput.addEventListener("keydown", (event) => {
    const results = [...el.placeResults.querySelectorAll(".result")];
    if (!results.length) return;
    if (event.key === "ArrowDown") { event.preventDefault(); results[0].focus(); }
    if (event.key === "Enter") { event.preventDefault(); results[0].click(); }
  });

  el.placeResults.addEventListener("keydown", (event) => {
    const results = [...el.placeResults.querySelectorAll(".result")];
    const current = results.indexOf(document.activeElement);
    if (event.key === "ArrowDown") {
      event.preventDefault();
      if (current + 1 < results.length) results[current + 1].focus();
    }
    if (event.key === "ArrowUp") {
      event.preventDefault();
      if (current > 0) results[current - 1].focus();
      else el.placeInput.focus();
    }
  });

  /* Offered, never taken. The browser only prompts when this is clicked, so
     a person who never wants to share a position is never asked. */
  async function useDeviceLocation() {
    if (!navigator.geolocation) {
      say(t().locationUnavailable, "warn");
      return;
    }
    el.placeHint.hidden = false;
    el.placeHint.textContent = t().locating;
    navigator.geolocation.getCurrentPosition(
      async (position) => {
        const { latitude, longitude } = position.coords;
        let named = null;
        try {
          named = await nameThePoint(latitude, longitude);
        } catch {
          named = { coordinates: { latitude, longitude } };
        }
        setPlace(named, { announce: true });
        closePopover(el.placeButton, el.placePanel);
        el.placeButton.focus();
      },
      (error) => {
        el.placeHint.textContent =
          error && error.code === 1 ? t().locationDenied : t().locationUnavailable;
      },
      { enableHighAccuracy: false, timeout: 10000, maximumAge: 600000 }
    );
  }

  el.useLocation.addEventListener("click", useDeviceLocation);

  async function runPlaceSearch(query) {
    const token = ++searchToken;
    let results;
    try {
      results = await searchLocations(query, 8);
    } catch {
      if (token !== searchToken) return;
      el.placeResults.textContent = "";
      el.placeHint.textContent = t().placeSearchFailed;
      return;
    }
    if (token !== searchToken) return;

    el.placeResults.textContent = "";
    if (!results.length) {
      el.placeHint.textContent = t().noPlaces(query);
      el.placeHint.hidden = false;
      el.placeInput.setAttribute("aria-expanded", "false");
      return;
    }
    el.placeHint.hidden = true;
    el.placeInput.setAttribute("aria-expanded", "true");

    for (const candidate of results) {
      const option = make("button", "result");
      option.type = "button";
      option.setAttribute("role", "option");
      option.setAttribute("aria-selected", "false");
      option.appendChild(make("span", "result__name", candidate.name || displayName(candidate)));
      const where = [candidate.admin1, candidate.country].filter(Boolean).join(", ");
      if (where) option.appendChild(make("span", "result__where", where));
      const c = candidate.coordinates || {};
      option.appendChild(make("span", "result__coords",
        `${Number(c.latitude).toFixed(2)}, ${Number(c.longitude).toFixed(2)}`));
      option.addEventListener("click", () => {
        setPlace(candidate, { announce: true });
        closePopover(el.placeButton, el.placePanel);
        el.placeButton.focus();
      });
      el.placeResults.appendChild(option);
    }
  }

  /* ------------------------------------------------------- speech to text */

  const SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition || null;
  let recognition = null;

  function recognitionError(code) {
    switch (code) {
      case "not-allowed":
      case "service-not-allowed":
        return "Microphone permission is blocked. Allow microphone access in the browser, then try again — or type your question.";
      case "no-speech":
        return "Nothing was picked up. Try again, a little closer to the microphone.";
      case "audio-capture":
        return "No microphone was found. Connect one, or type your question.";
      case "network":
        return "Speech recognition needs the network and couldn’t reach it. Typing still works.";
      case "aborted":
        return "";
      default:
        return "Speech recognition stopped unexpectedly. You can type your question instead.";
    }
  }

  function setListening(listening) {
    state.listening = listening;
    const target = state.micMode === "ask" ? el.speak : el.dictate;
    el.speak.dataset.listening = String(listening && state.micMode === "ask");
    el.dictate.dataset.listening = String(listening && state.micMode === "dictate");
    el.speak.classList.toggle("pulse", listening && state.micMode === "ask");
    el.dictate.classList.toggle("pulse", listening && state.micMode === "dictate");
    const label = el.speak.querySelector(".btn__label");
    if (label) label.textContent = listening && state.micMode === "ask" ? "Listening" : "Speak";
    el.speak.setAttribute("aria-pressed", String(listening && state.micMode === "ask"));
    if (!listening && target === el.speak) el.speak.disabled = state.busy;
  }

  function startListening(mode) {
    if (!SpeechRecognitionAPI) {
      say(t().noVoiceInput, "warn");
      return;
    }
    if (state.listening) { if (recognition) recognition.abort(); return; }

    state.micMode = mode;
    try {
      recognition = new SpeechRecognitionAPI();
    } catch (error) {
      say("The microphone could not be started. Typing still works.", "error");
      return;
    }
    recognition.lang = state.language.tag;
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;
    recognition.continuous = false;

    recognition.onstart = () => {
      setListening(true);
      say(mode === "ask" ? t().listening : t().dictating);
    };
    recognition.onresult = (event) => {
      const transcript = (event.results[0][0].transcript || "").trim();
      if (!transcript) return;
      if (mode === "ask") {
        ask(transcript);
      } else {
        el.prompt.value = el.prompt.value ? `${el.prompt.value} ${transcript}` : transcript;
        autoGrow();
        updateSend();
        el.prompt.focus();
        say("");
      }
    };
    recognition.onerror = (event) => {
      const message = recognitionError(event.error);
      if (message) say(message, "error");
    };
    recognition.onend = () => setListening(false);

    try {
      recognition.start();
    } catch (error) {
      setListening(false);
      say("The microphone could not be started. Typing still works.", "error");
    }
  }

  el.speak.addEventListener("click", () => startListening("ask"));
  el.dictate.addEventListener("click", () => startListening("dictate"));

  /* ------------------------------------------------------- text to speech */

  // The provenance block is for reading, not listening.
  function spokenPart(answer) {
    const cut = answer.search(/^(Sources and freshness:|स्रोत और नवीनता:)/m);
    return (cut === -1 ? answer : answer.slice(0, cut)).trim();
  }

  function speak(answer, responseLanguage) {
    state.lastSpoken = spokenPart(answer);
    el.replay.disabled = !state.lastSpoken;
    if (!("speechSynthesis" in window)) {
      say(t().noVoiceOutput, "warn");
      return;
    }
    try {
      window.speechSynthesis.cancel();
      const tag = String(responseLanguage).includes("-")
        ? responseLanguage
        : (String(responseLanguage) === "hi" ? "hi-IN" : "en-IN");
      const utterance = new SpeechSynthesisUtterance(state.lastSpoken);
      utterance.lang = tag;
      const voice = voiceFor(tag);
      if (voice) utterance.voice = voice;
      else {
        const named = LANGUAGES.find((l) => l.tag === tag) || state.language;
        say(t().noVoiceFor(named.native), "warn");
      }
      utterance.onend = () => { el.stop.disabled = true; };
      utterance.onerror = () => { el.stop.disabled = true; };
      el.stop.disabled = false;
      window.speechSynthesis.speak(utterance);
    } catch (error) {
      say(t().noVoiceOutput, "warn");
    }
  }

  el.replay.addEventListener("click", () => {
    if (state.lastSpoken) speak(state.lastSpoken, state.language.tag);
  });
  el.stop.addEventListener("click", () => {
    if (window.speechSynthesis) window.speechSynthesis.cancel();
    el.stop.disabled = true;
  });

  /* ---------------------------------------------------------------- start */

  function boot() {
    const savedLanguage = recall(STORE_LANG);
    selectLanguage(LANGUAGES.some((l) => l.tag === savedLanguage) ? savedLanguage : LANGUAGES[0].tag);
    say("");

    const savedPlace = recall(STORE_PLACE);
    if (savedPlace && savedPlace.coordinates && savedPlace.name) setPlace(savedPlace);
    else renderChips();

    renderLanguageMenu();
    el.placeHint.textContent = t().typeMore;
    fitPlaceholder();
    updateSend();

    if (!SpeechRecognitionAPI) {
      el.speak.disabled = true;
      el.dictate.disabled = true;
      el.speak.title = t().noVoiceInput;
      el.dictate.title = t().noVoiceInput;
      say(t().noVoiceInput, "warn");
    }
    if (window.speechSynthesis) {
      window.speechSynthesis.onvoiceschanged = () => renderLanguageMenu();
    }
    // The badge has to be right before the panel is opened, so watches are
    // read once at start rather than only on first open.
    loadWatches();
    startWatchPolling();
  }

  /* ------------------------------------------------------------- alerts --

     Watched locations. This panel is a view onto the backend's deterministic
     alert engine: it saves where to watch, asks the API what stands there, and
     renders exactly the fields that come back. Every severity, threshold and
     title below was decided by a rule on the server against validated data.
     Nothing here judges whether weather is dangerous, and nothing here may. */

  const SUBSCRIPTIONS_URL = "/api/v1/alerts/subscriptions";

  /* Identifies this browser's set of watches. Not an account and not a
     credential — it authorises nothing and carries no personal data. It exists
     so a demo needs no sign-in; a real deployment swaps it for its own subject. */
  function clientKey() {
    let key = recall(STORE_CLIENT);
    if (typeof key === "string" && key.length >= 8) return key;
    key = "c-" + (crypto.randomUUID ? crypto.randomUUID() : String(Date.now()) + Math.random().toString(36).slice(2));
    remember(STORE_CLIENT, key);
    return key;
  }

  async function watchApi(path, options) {
    const settings = options || {};
    const response = await fetch(SUBSCRIPTIONS_URL + (path || ""), {
      method: settings.method || "GET",
      headers: {
        Accept: "application/json",
        "X-WeatherGPT-Client": clientKey(),
        ...(settings.body ? { "Content-Type": "application/json" } : {}),
      },
      body: settings.body ? JSON.stringify(settings.body) : undefined,
    });
    if (response.status === 204) return null;
    let body = null;
    try { body = await response.json(); } catch { body = null; }
    if (!response.ok) {
      const error = new Error("watch " + response.status);
      error.status = response.status;
      error.code = body && body.error ? body.error.code : null;
      error.detail = body && body.error ? body.error : null;
      throw error;
    }
    return body;
  }

  const watchState = { items: [], loaded: false, pending: null, busy: false };

  /* A backend failure becomes one sentence a person can act on, never a status
     code and never a stack trace. */
  function watchProblem(error) {
    if (error && error.code === "DATABASE_UNAVAILABLE") return t().watchNeedsDb;
    if (error && error.code === "AMBIGUOUS_LOCATION") return t().watchAmbiguous;
    return t().watchFailed;
  }

  function setWatchHint(message, hidden) {
    el.watchHint.hidden = Boolean(hidden);
    if (message) el.watchHint.textContent = message;
  }

  async function loadWatches(options) {
    const settings = options || {};
    try {
      const body = await watchApi("");
      watchState.items = body && Array.isArray(body.subscriptions) ? body.subscriptions : [];
      watchState.loaded = true;
      renderWatches();
    } catch (error) {
      watchState.items = [];
      watchState.loaded = true;
      renderWatches(watchProblem(error));
      if (settings.announce) say(watchProblem(error), "warn");
    }
  }

  /* The badge counts hazards, not watched places: a number on an alert bell
     that means "locations" would read as "three alerts" at a glance. */
  function renderWatchBadge() {
    let total = 0;
    for (const item of watchState.items) {
      if (item.enabled && Array.isArray(item.alerts)) total += item.alerts.length;
    }
    el.alertsCount.hidden = total === 0;
    el.alertsCount.textContent = String(total);
    el.alertsButton.classList.toggle("pill--live", total > 0);
  }

  function hazardCard(alert) {
    const severity = String(alert.severity || "").toLowerCase();
    const card = make("div", `hazard hazard--${severity || "info"}`);

    const top = make("div", "hazard__top");
    top.appendChild(make("span", "hazard__title", alert.title || alert.alert_type));
    if (severity) top.appendChild(make("span", "hazard__sev", severity));
    card.appendChild(top);

    if (alert.description) card.appendChild(make("p", "hazard__body", alert.description));

    // Evidence: the measured value, the threshold it crossed, and when. This is
    // the reason the rule fired, and it is the difference between an alert a
    // person can check and one they must simply believe.
    const meta = make("div", "hazard__meta");
    const evidence = alert.evidence || {};
    if (evidence.observed_value != null) {
      const unit = evidence.unit ? ` ${evidence.unit}` : "";
      const measured = make("span");
      measured.appendChild(make("b", null, "Measured "));
      measured.appendChild(document.createTextNode(`${readable(g(evidence.observed_value)).text}${unit}`));
      meta.appendChild(measured);
    }
    if (evidence.threshold != null) {
      const unit = evidence.unit ? ` ${evidence.unit}` : "";
      const limit = make("span");
      limit.appendChild(make("b", null, "Threshold "));
      limit.appendChild(document.createTextNode(`${readable(g(evidence.threshold)).text}${unit}`));
      meta.appendChild(limit);
    }
    if (alert.valid_until) {
      const until = make("span");
      until.appendChild(make("b", null, "Until "));
      until.appendChild(document.createTextNode(formatStamp(alert.valid_until)));
      meta.appendChild(until);
    }
    if (meta.childNodes.length) card.appendChild(meta);

    // The source that produced the number the rule read.
    const provenance = alert.provenance || {};
    if (provenance.provider_name) {
      const source = make("div", "hazard__meta");
      const line = make("span");
      line.appendChild(make("b", null, "Source "));
      line.appendChild(document.createTextNode(provenance.provider_name));
      source.appendChild(line);
      card.appendChild(source);
    }
    return card;
  }

  function watchCard(item) {
    const row = make("div", "watched");

    const top = make("div", "watched__top");
    const place = make("div", "watched__place");
    place.appendChild(document.createTextNode(item.location.name || "—"));
    const where = [item.location.admin1, item.location.country].filter(Boolean).join(", ");
    if (where) place.appendChild(make("div", "watched__where", where));
    top.appendChild(place);

    const actions = make("div", "watched__actions");

    const refresh = make("button", "watched__action");
    refresh.type = "button";
    refresh.title = "Check now";
    refresh.setAttribute("aria-label", `Check ${item.location.name} now`);
    refresh.appendChild(icon("ic-replay"));
    refresh.addEventListener("click", () => refreshWatch(item.id));
    actions.appendChild(refresh);

    const toggle = make("button", "watched__action");
    toggle.type = "button";
    toggle.title = item.enabled ? "Pause monitoring" : "Resume monitoring";
    toggle.setAttribute("aria-label", toggle.title + ` for ${item.location.name}`);
    toggle.appendChild(icon(item.enabled ? "ic-stop" : "ic-check"));
    toggle.addEventListener("click", () => toggleWatch(item.id, !item.enabled));
    actions.appendChild(toggle);

    const remove = make("button", "watched__action watched__action--off");
    remove.type = "button";
    remove.title = "Stop watching";
    remove.setAttribute("aria-label", `Stop watching ${item.location.name}`);
    remove.appendChild(icon("ic-offline"));
    remove.addEventListener("click", () => removeWatch(item.id));
    actions.appendChild(remove);

    top.appendChild(actions);
    row.appendChild(top);

    // Three distinct facts: watching and checked, watching but not yet checked,
    // and paused. Showing the middle one as "no alerts" would be an all-clear
    // nobody has actually verified.
    const stateLine = make("div", "watched__state");
    let dot = "watched__dot";
    let label;
    if (!item.enabled) {
      dot += " watched__dot--idle";
      label = t().watchPaused;
    } else if (!item.evaluated) {
      dot += " watched__dot--waiting";
      label = t().watchPending;
    } else {
      label = item.last_evaluated_at
        ? `${t().watchMonitoring} · ${t().watchChecked(formatTime(item.last_evaluated_at))}`
        : t().watchMonitoring;
    }
    stateLine.appendChild(make("span", dot));
    stateLine.appendChild(make("span", null, label));
    row.appendChild(stateLine);

    if (item.enabled) {
      const alerts = Array.isArray(item.alerts) ? item.alerts : [];
      if (alerts.length) {
        const list = make("div", "watched__alerts");
        for (const alert of alerts) list.appendChild(hazardCard(alert));
        row.appendChild(list);
      } else if (item.evaluated) {
        const clear = make("div", "watch__clear");
        clear.appendChild(icon("ic-shield"));
        clear.appendChild(make("span", null, t().watchNoAlerts));
        row.appendChild(clear);
      }
    }
    return row;
  }

  function renderWatches(problem) {
    el.watchList.textContent = "";
    if (problem) {
      el.watchList.appendChild(make("p", "watch__empty", problem));
      renderWatchBadge();
      return;
    }
    if (!watchState.items.length) {
      el.watchList.appendChild(make("p", "watch__empty", t().watchEmpty));
      renderWatchBadge();
      return;
    }
    for (const item of watchState.items) el.watchList.appendChild(watchCard(item));
    renderWatchBadge();
  }

  /* --- adding a watch: search, confirm, then create ---------------------- */

  let watchSearchTimer = null;
  let watchSearchToken = 0;

  function clearWatchConfirm() {
    watchState.pending = null;
    el.watchConfirm.hidden = true;
  }

  async function runWatchSearch(query) {
    const token = ++watchSearchToken;
    let results;
    try {
      results = await searchLocations(query, 6);
    } catch {
      if (token !== watchSearchToken) return;
      el.watchResults.textContent = "";
      setWatchHint(t().placeSearchFailed, false);
      return;
    }
    if (token !== watchSearchToken) return;

    el.watchResults.textContent = "";
    if (!results.length) {
      setWatchHint(t().noPlaces(query), false);
      el.watchInput.setAttribute("aria-expanded", "false");
      return;
    }
    setWatchHint(null, true);
    el.watchInput.setAttribute("aria-expanded", "true");

    for (const candidate of results) {
      const option = make("button", "result");
      option.type = "button";
      option.setAttribute("role", "option");
      option.setAttribute("aria-selected", "false");
      option.appendChild(make("span", "result__name", candidate.name || displayName(candidate)));
      const where = [candidate.admin1, candidate.country].filter(Boolean).join(", ");
      if (where) option.appendChild(make("span", "result__where", where));
      // Picking a candidate does not start the watch. It fills the confirm
      // step, so the place is read back before anything is saved.
      option.addEventListener("click", () => {
        watchState.pending = candidate;
        el.watchConfirmPlace.textContent = displayName(candidate);
        el.watchConfirm.hidden = false;
        el.watchResults.textContent = "";
        el.watchInput.value = "";
        el.watchInput.setAttribute("aria-expanded", "false");
        el.watchCreate.focus();
      });
      el.watchResults.appendChild(option);
    }
  }

  async function createWatch() {
    const candidate = watchState.pending;
    if (!candidate || watchState.busy) return;
    const coordinates = candidate.coordinates || {};
    watchState.busy = true;
    el.watchCreate.disabled = true;
    try {
      // Coordinates, not the typed text: a watch must not drift onto a
      // different place of the same name later.
      await watchApi("", {
        method: "POST",
        body: {
          latitude: coordinates.latitude,
          longitude: coordinates.longitude,
          name: candidate.name || displayName(candidate),
          admin1: candidate.admin1 || null,
          country: candidate.country || null,
          timezone: candidate.timezone || null,
          alert_types: [],
        },
      });
      clearWatchConfirm();
      await loadWatches();
      say(t().watchAdded(displayName(candidate)), "ok");
    } catch (error) {
      setWatchHint(watchProblem(error), false);
    } finally {
      watchState.busy = false;
      el.watchCreate.disabled = false;
    }
  }

  async function refreshWatch(id) {
    try {
      await watchApi(`/${id}/refresh`, { method: "POST" });
      await loadWatches();
    } catch (error) {
      renderWatches(watchProblem(error));
    }
  }

  async function toggleWatch(id, enabled) {
    try {
      await watchApi(`/${id}`, { method: "PATCH", body: { enabled } });
      await loadWatches();
    } catch (error) {
      renderWatches(watchProblem(error));
    }
  }

  async function removeWatch(id) {
    try {
      await watchApi(`/${id}`, { method: "DELETE" });
      await loadWatches();
      say(t().watchRemoved, "ok");
    } catch (error) {
      renderWatches(watchProblem(error));
    }
  }

  el.watchInput.addEventListener("input", () => {
    const query = el.watchInput.value.trim();
    window.clearTimeout(watchSearchTimer);
    if (query.length < 2) {
      el.watchResults.textContent = "";
      el.watchInput.setAttribute("aria-expanded", "false");
      setWatchHint(null, false);
      return;
    }
    watchSearchTimer = window.setTimeout(() => runWatchSearch(query), 280);
  });

  el.watchCreate.addEventListener("click", createWatch);
  el.watchCancel.addEventListener("click", () => {
    clearWatchConfirm();
    el.watchInput.focus();
  });

  el.alertsButton.addEventListener("click", () => {
    togglePopover(el.alertsButton, el.alertsPanel);
    if (!el.alertsPanel.hidden) {
      // Always re-read on open: the monitor may have swept since last time.
      loadWatches({ announce: false });
      el.watchInput.focus();
    }
  });

  /* Kept fresh while the tab is open, so a hazard that appears between sweeps
     shows up without the person reopening the panel. Cheap: one small read. */
  function startWatchPolling() {
    window.setInterval(() => {
      if (document.hidden) return;
      loadWatches();
    }, 120000);
  }

  boot();

  // Lets an automated check drive the pipeline without a microphone.
  window.__voice = {
    handleTranscript: (text) => ask(text),
    setLanguage: (tag) => selectLanguage(tag),
    spoken: () => state.lastSpoken,
    watches: () => watchState.items,
    reloadWatches: () => loadWatches(),
  };
})();
