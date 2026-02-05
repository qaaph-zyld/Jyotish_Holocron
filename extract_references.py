"""
Chart-to-Textbook Reference Extraction Pipeline
=================================================
Hybrid approach:
  Pass 1: Deterministic grep on vedic_astro_textbook.md
  Pass 2: PageIndex RAG (LLM-reasoned) gap-filling from vedic_astro_textbook.pdf

Usage: python extract_references.py
Output: chart_reference_report.md
"""

import re
import os
import sys
import json
import time
from collections import OrderedDict

# ── Configuration ──────────────────────────────────────────────────────
CHART_FILE = "Nikola_Jelacic.txt"
TEXTBOOK_MD = "vedic_astro_textbook.md"
TEXTBOOK_PDF = "vedic_astro_textbook.pdf"
REPORT_FILE = "chart_reference_report.md"
CONTEXT_LINES = 5  # lines above/below each grep hit

# PageIndex RAG imports (deferred to avoid crash if not needed)
PAGEINDEX_AVAILABLE = False
try:
    from chat_pageindex import (
        BOOK_TREE, PDF_PAGE_OFFSET,
        tree_search, resolve_nodes, extract_pages,
        score_page_relevance, call_ollama, build_tree_for_prompt,
    )
    from pypdf import PdfReader
    PAGEINDEX_AVAILABLE = True
except ImportError:
    pass

# Preferred Ollama model for RAG pass
RAG_MODEL = "qwen2.5:7b"
RAG_MODEL_FALLBACK = "qwen2.5:3b"

# ── Astrological Reference Data ───────────────────────────────────────

RASI_FULL = {
    "Ar": "Aries", "Ta": "Taurus", "Ge": "Gemini", "Cn": "Cancer",
    "Le": "Leo", "Vi": "Virgo", "Li": "Libra", "Sc": "Scorpio",
    "Sg": "Sagittarius", "Cp": "Capricorn", "Aq": "Aquarius", "Pi": "Pisces",
}

RASI_SANSKRIT = {
    "Ar": "Mesha", "Ta": "Vrishabha", "Ge": "Mithuna", "Cn": "Karkataka",
    "Le": "Simha", "Vi": "Kanya", "Li": "Tula", "Sc": "Vrischika",
    "Sg": "Dhanu", "Cp": "Makara", "Aq": "Kumbha", "Pi": "Meena",
}

NAKSHATRA_FULL = {
    "Aswi": "Aswini", "Bhar": "Bharani", "Krit": "Krittika",
    "Rohi": "Rohini", "Mrig": "Mrigasira", "Ardr": "Ardra",
    "Puna": "Punarvasu", "Push": "Pushyami", "Asre": "Aslesha",
    "Magh": "Magha", "PPha": "Purva Phalguni", "UPha": "Uttara Phalguni",
    "Hast": "Hasta", "Chit": "Chitra", "Swat": "Swati",
    "Visa": "Visakha", "Anu": "Anuradha", "Jye": "Jyeshtha",
    "Mool": "Moola", "PSha": "Purva Ashadha", "USha": "Uttara Ashadha",
    "Srav": "Sravana", "Dhan": "Dhanishta", "Sata": "Satabhisha",
    "PBha": "Purva Bhadrapada", "UBha": "Uttara Bhadrapada", "Reva": "Revati",
}

PLANET_NAMES = ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu", "Ketu"]

# Lordship from Libra lagna (sign number → lord)
# Li=1, Sc=2, Sg=3, Cp=4, Aq=5, Pi=6, Ar=7, Ta=8, Ge=9, Cn=10, Le=11, Vi=12
HOUSE_LORDS_LIBRA = {
    1: "Venus", 2: "Mars", 3: "Jupiter", 4: "Saturn",
    5: "Saturn", 6: "Jupiter", 7: "Mars", 8: "Venus",
    9: "Mercury", 10: "Moon", 11: "Sun", 12: "Mercury",
}

HOUSE_SIGNS_LIBRA = {
    1: "Li", 2: "Sc", 3: "Sg", 4: "Cp", 5: "Aq", 6: "Pi",
    7: "Ar", 8: "Ta", 9: "Ge", 10: "Cn", 11: "Le", 12: "Vi",
}

DIVISIONAL_CHARTS = {
    "D-1": "Rasi", "D-2": "Hora", "D-3": "Drekkana",
    "D-4": "Chaturthamsa", "D-5": "Panchamsa", "D-6": "Shashthamsa",
    "D-7": "Saptamsa", "D-8": "Ashtamsa", "D-9": "Navamsa",
    "D-10": "Dasamsa", "D-11": "Rudramsa", "D-12": "Dwadasamsa",
    "D-16": "Shodasamsa", "D-20": "Vimsamsa", "D-24": "Siddhamsa",
    "D-27": "Nakshatramsa", "D-30": "Trimsamsa", "D-40": "Khavedamsa",
    "D-45": "Akshavedamsa", "D-60": "Shashtyamsa",
    "D-81": "Navanavamsa", "D-108": "Ashtottaramsa",
    "D-144": "Dwadasa-Dwadasamsa",
}

DASA_SYSTEMS = [
    "Vimsottari", "Ashtottari", "Kalachakra", "Narayana",
    "Sudasa", "Moola",
]

# ── Phase A: Parse Chart & Generate Terms ──────────────────────────────

def parse_chart(filepath):
    """Parse Nikola_Jelacic.txt and extract structured chart data."""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    lines = text.splitlines()

    data = {
        "planets": {},       # planet -> {rasi, nak, pada, navamsa, karaka}
        "lagna": {},         # rasi, nak, pada, navamsa
        "chara_karakas": {}, # karaka_abbr -> {planet, meaning}
        "special_lagnas": [],
        "upagrahas": [],
        "avasthas": {},      # planet -> {age, alertness, mood, activity}
    }

    # Parse planet table (lines like: "Sun - BK   22 Vi 04' 06.58" Hast 4  Vi  Cn")
    planet_re = re.compile(
        r"^(Sun|Moon|Mars|Mercury|Jupiter|Venus|Saturn|Rahu|Ketu)\s*-?\s*(\w*)\s+"
        r"(\d+)\s+(\w{2})\s+\d+'\s+[\d.]+\"\s+(\w+)\s+(\d+)\s+(\w{2})\s+(\w{2})"
    )
    lagna_re = re.compile(
        r"^Lagna\s+(\d+)\s+(\w{2})\s+\d+'\s+[\d.]+\"\s+(\w+)\s+(\d+)\s+(\w{2})\s+(\w{2})"
    )
    special_re = re.compile(
        r"^(Bhava Lagna|Hora Lagna|Ghati Lagna|Vighati Lagna|Varnada Lagna|"
        r"Sree Lagna|Pranapada Lagna|Indu Lagna|Bhrigu Bindu|Maandi|Gulika)\s+"
        r"(\d+)\s+(\w{2})"
    )
    karaka_re = re.compile(
        r"^(AK|AmK|BK|MK|PiK|PK|GK|DK)\s+(Sun|Moon|Mars|Mercury|Jupiter|Venus|Saturn|Rahu|Ketu)\s+(.*)"
    )

    for line in lines:
        line = line.strip()

        m = lagna_re.match(line)
        if m:
            data["lagna"] = {
                "degree": m.group(1), "rasi": m.group(2),
                "nak": m.group(3), "pada": m.group(4),
                "rasi_sign": m.group(5), "navamsa": m.group(6),
            }
            continue

        m = planet_re.match(line)
        if m:
            planet = m.group(1)
            data["planets"][planet] = {
                "karaka": m.group(2),
                "degree": m.group(3), "rasi": m.group(4),
                "nak": m.group(5), "pada": m.group(6),
                "rasi_sign": m.group(7), "navamsa": m.group(8),
            }
            continue

        m = special_re.match(line)
        if m:
            name = m.group(1)
            if name in ("Maandi", "Gulika"):
                data["upagrahas"].append({"name": name, "rasi": m.group(3)})
            else:
                data["special_lagnas"].append({"name": name, "rasi": m.group(3)})
            continue

        m = karaka_re.match(line)
        if m:
            data["chara_karakas"][m.group(1)] = {
                "planet": m.group(2), "meaning": m.group(3).strip()
            }
            continue

    # Parse avasthas
    avastha_section = False
    current_field = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Planet") and "Age" in stripped:
            current_field = "age_alert_mood"
            continue
        if stripped.startswith("Planet") and "Activity" in stripped:
            current_field = "activity"
            continue
        if current_field == "age_alert_mood":
            for planet in PLANET_NAMES:
                if stripped.startswith(planet):
                    parts = stripped.split()
                    if planet not in data["avasthas"]:
                        data["avasthas"][planet] = {}
                    # Extract parenthesized terms
                    parens = re.findall(r'\((\w+)\)', stripped)
                    data["avasthas"][planet]["avastha_raw"] = stripped
                    data["avasthas"][planet]["avastha_terms"] = parens
        if current_field == "activity":
            for planet in PLANET_NAMES:
                if stripped.startswith(planet):
                    if planet not in data["avasthas"]:
                        data["avasthas"][planet] = {}
                    parens = re.findall(r'\((\w+)\)', stripped)
                    data["avasthas"][planet]["activity_terms"] = parens

    return data


def generate_terms(chart_data):
    """Generate categorized searchable terms from parsed chart data."""
    terms = OrderedDict()  # category -> list of (term, context_note)

    # ── 1. Rasi (planet-in-sign) ──
    rasi_terms = []
    for planet, info in chart_data["planets"].items():
        rasi_abbr = info["rasi"]
        rasi_full = RASI_FULL.get(rasi_abbr, rasi_abbr)
        rasi_skt = RASI_SANSKRIT.get(rasi_abbr, "")
        rasi_terms.append((rasi_full, f"{planet} is in {rasi_full}"))
        if rasi_skt:
            rasi_terms.append((rasi_skt, f"{planet} in {rasi_full} (Sanskrit: {rasi_skt})"))
        rasi_terms.append((f"{planet}", f"planet: {planet}"))
    # Lagna sign
    lg = chart_data["lagna"]
    if lg:
        rasi_terms.append((RASI_FULL.get(lg["rasi"], lg["rasi"]), f"Lagna in {RASI_FULL.get(lg['rasi'], lg['rasi'])}"))
        rasi_terms.append((RASI_SANSKRIT.get(lg["rasi"], ""), f"Lagna (Sanskrit)"))
        rasi_terms.append(("Libra lagna", "Ascendant is Libra"))
        rasi_terms.append(("Tula lagna", "Ascendant is Tula/Libra"))
    terms["Rasi (Signs & Planets-in-Signs)"] = rasi_terms

    # ── 2. Nakshatras ──
    nak_terms = []
    seen_naks = set()
    for planet, info in chart_data["planets"].items():
        nak_abbr = info["nak"]
        nak_full = NAKSHATRA_FULL.get(nak_abbr, nak_abbr)
        if nak_full not in seen_naks:
            nak_terms.append((nak_full, f"{planet}'s nakshatra"))
            seen_naks.add(nak_full)
    if lg:
        nak_full = NAKSHATRA_FULL.get(lg["nak"], lg["nak"])
        if nak_full not in seen_naks:
            nak_terms.append((nak_full, f"Lagna nakshatra"))
            seen_naks.add(nak_full)
    nak_terms.append(("nakshatra", "general concept"))
    terms["Nakshatras"] = nak_terms

    # ── 3. Houses & Lordships ──
    house_terms = []
    for house_num, lord in HOUSE_LORDS_LIBRA.items():
        ordinal = {1:"1st",2:"2nd",3:"3rd",4:"4th",5:"5th",6:"6th",
                   7:"7th",8:"8th",9:"9th",10:"10th",11:"11th",12:"12th"}[house_num]
        house_terms.append((f"lord of {ordinal}", f"{lord} rules {ordinal} house from Libra"))
        house_terms.append((f"{ordinal} house", f"house {house_num}"))
        house_terms.append((f"{ordinal} lord", f"{lord}"))
    # Functional nature terms
    house_terms.append(("kendra", "houses 1,4,7,10"))
    house_terms.append(("trikona", "houses 1,5,9"))
    house_terms.append(("dusthana", "houses 6,8,12"))
    house_terms.append(("maraka", "houses 2,7"))
    house_terms.append(("upachaya", "houses 3,6,10,11"))
    house_terms.append(("house lord", "general concept"))
    terms["Houses & Lordships"] = house_terms

    # ── 4. Chara Karakas ──
    karaka_terms = []
    karaka_full_names = {
        "AK": "Atmakaraka", "AmK": "Amatyakaraka", "BK": "Bhratrikaraka",
        "MK": "Matrikaraka", "PiK": "Pitrikaraka", "PK": "Putrakaraka",
        "GK": "Gnatikaraka", "DK": "Darakaraka",
    }
    for abbr, info in chart_data["chara_karakas"].items():
        full = karaka_full_names.get(abbr, abbr)
        karaka_terms.append((full, f"{full} = {info['planet']} ({info['meaning']})"))
        karaka_terms.append((abbr, f"abbreviation for {full}"))
    karaka_terms.append(("chara karaka", "variable significator system"))
    karaka_terms.append(("sthira karaka", "fixed significator system"))
    karaka_terms.append(("naisargika karaka", "natural significator"))
    terms["Chara Karakas"] = karaka_terms

    # ── 5. Special Lagnas ──
    sl_terms = []
    for sl in chart_data["special_lagnas"]:
        sl_terms.append((sl["name"], f"in {RASI_FULL.get(sl['rasi'], sl['rasi'])}"))
    sl_terms.append(("special lagna", "general concept"))
    sl_terms.append(("Arudha Lagna", "AL — image of self"))
    sl_terms.append(("arudha pada", "general concept"))
    sl_terms.append(("upapada", "arudha of 12th house"))
    terms["Special Lagnas & Arudhas"] = sl_terms

    # ── 6. Upagrahas ──
    upa_terms = []
    for u in chart_data["upagrahas"]:
        upa_terms.append((u["name"], f"in {RASI_FULL.get(u['rasi'], u['rasi'])}"))
    upa_terms.append(("upagraha", "sub-planets"))
    upa_terms.append(("Dhuma", "upagraha"))
    upa_terms.append(("Vyatipata", "upagraha"))
    terms["Upagrahas"] = upa_terms

    # ── 7. Ashtakavarga ──
    ak_terms = [
        ("ashtakavarga", "bindus system"),
        ("sarvashtakavarga", "combined ashtakavarga"),
        ("bhinnashtakavarga", "individual planet ashtakavarga"),
        ("Sodhya Pinda", "rectified strength total"),
        ("Rasi Pinda", "sign-based pinda"),
        ("Graha Pinda", "planet-based pinda"),
        ("kaksha", "sub-division within ashtakavarga"),
        ("bindu", "benefic point in ashtakavarga"),
    ]
    terms["Ashtakavarga"] = ak_terms

    # ── 8. Strength Systems ──
    str_terms = [
        ("shadbala", "sixfold strength"),
        ("Ishta Phala", "desired results strength"),
        ("Kashta Phala", "suffering strength"),
        ("vimsopaka", "20-point strength"),
        ("vaiseshikamsa", "special divisional dignity"),
        ("Paarijaata", "vaiseshikamsa level"),
        ("Uttama", "vaiseshikamsa level"),
        ("Gopura", "vaiseshikamsa level"),
        ("Simhasana", "vaiseshikamsa level"),
        ("Kerala", "vaiseshikamsa level"),
        ("Kalpavriksha", "vaiseshikamsa level"),
        ("Kanduka", "vaiseshikamsa level"),
        ("Kusuma", "vaiseshikamsa level"),
    ]
    terms["Strength (Shadbala, Vimsopaka, Vaiseshikamsa)"] = str_terms

    # ── 9. Avasthas ──
    av_terms = []
    seen_av = set()
    for planet, info in chart_data["avasthas"].items():
        for t in info.get("avastha_terms", []) + info.get("activity_terms", []):
            if t not in seen_av:
                av_terms.append((t, f"avastha of {planet}"))
                seen_av.add(t)
    av_terms.append(("avastha", "planetary state"))
    av_terms.append(("bala avastha", "infant state"))
    av_terms.append(("kumara avastha", "adolescent state"))
    av_terms.append(("yuva avastha", "young state"))
    av_terms.append(("mrita avastha", "dead state"))
    av_terms.append(("vriddha avastha", "old state"))
    terms["Avasthas (Planetary States)"] = av_terms

    # ── 10. Divisional Charts ──
    div_terms = []
    for code, name in DIVISIONAL_CHARTS.items():
        div_terms.append((name, f"divisional chart {code}"))
        div_terms.append((code, f"{name}"))
    div_terms.append(("divisional chart", "general concept"))
    div_terms.append(("varga", "divisional chart system"))
    terms["Divisional Charts"] = div_terms

    # ── 11. Dasa Systems ──
    dasa_terms = []
    for d in DASA_SYSTEMS:
        dasa_terms.append((d, f"dasa system"))
    dasa_terms.append(("dasa", "planetary period system"))
    dasa_terms.append(("antardasa", "sub-period"))
    dasa_terms.append(("mahadasa", "major period"))
    dasa_terms.append(("pratyantardasa", "sub-sub-period"))
    terms["Dasa Systems"] = dasa_terms

    # ── 12. Planetary Status (exaltation, debilitation, etc.) ──
    status_terms = [
        ("exaltation", "planet at peak strength"),
        ("exalted", "planet in exaltation sign"),
        ("debilitation", "planet at weakest"),
        ("debilitated", "planet in debilitation sign"),
        ("own sign", "planet in its own rasi"),
        ("own house", "planet in its own rasi"),
        ("retrograde", "apparent backward motion"),
        ("combustion", "planet too close to Sun"),
        ("combust", "planet burned by Sun"),
        ("Jupiter in Cancer", "Jupiter exalted in chart"),
        ("Jupiter exalted", "Jupiter in Cancer = exaltation"),
    ]
    terms["Planetary Status"] = status_terms

    # ── 13. Yogas & Combinations ──
    yoga_terms = [
        ("yoga", "planetary combination"),
        ("raja yoga", "royal combination"),
        ("dhana yoga", "wealth combination"),
        ("Gajakesari", "Moon-Jupiter yoga"),
        ("Budhaditya", "Sun-Mercury yoga"),
        ("Pancha Mahapurusha", "5 great person yogas"),
        ("Hamsa yoga", "Jupiter in kendra in own/exaltation"),
        ("conjunction", "planets in same sign"),
        ("mutual aspect", "two planets aspecting each other"),
        ("benefic", "natural benefic planet"),
        ("malefic", "natural malefic planet"),
        ("functional benefic", "benefic for specific lagna"),
        ("functional malefic", "malefic for specific lagna"),
    ]
    terms["Yogas & Combinations"] = yoga_terms

    # ── 14. Aspects ──
    aspect_terms = [
        ("aspect", "planetary sight/drishti"),
        ("graha drishti", "planetary aspect"),
        ("rasi drishti", "sign aspect"),
        ("argala", "planetary intervention"),
        ("virodha argala", "obstructing intervention"),
        ("special aspect", "Mars/Jupiter/Saturn extra aspects"),
    ]
    terms["Aspects & Argalas"] = aspect_terms

    # ── 15. Transits & Timing ──
    transit_terms = [
        ("transit", "gochara"),
        ("Sade Sati", "Saturn transit over Moon"),
        ("gochara", "transit"),
        ("vedha", "transit obstruction"),
    ]
    terms["Transits & Timing"] = transit_terms

    return terms


# ── Phase B: Grep Search on .md ────────────────────────────────────────

def load_textbook(filepath):
    """Load the textbook markdown as a list of lines."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.readlines()


def grep_term(term, lines, context=CONTEXT_LINES):
    """Search for a term in textbook lines. Returns list of hits with context."""
    hits = []
    term_lower = term.lower()
    for i, line in enumerate(lines):
        if term_lower in line.lower():
            start = max(0, i - context)
            end = min(len(lines), i + context + 1)
            snippet = "".join(lines[start:end]).strip()
            hits.append({
                "line": i + 1,
                "snippet": snippet[:500],  # cap snippet length
            })
    return hits


def run_grep_pass(terms, lines):
    """Run grep for all terms. Returns results dict."""
    results = OrderedDict()
    total_terms = 0
    terms_with_hits = 0

    for category, term_list in terms.items():
        cat_results = []
        for term, note in term_list:
            if not term or len(term) < 2:
                continue
            total_terms += 1
            hits = grep_term(term, lines)
            if hits:
                terms_with_hits += 1
            cat_results.append({
                "term": term,
                "note": note,
                "hit_count": len(hits),
                "hits": hits[:10],  # cap at 10 hits per term for report size
            })
        results[category] = cat_results

    return results, total_terms, terms_with_hits


# ── Phase C: Report Generation ─────────────────────────────────────────

def generate_report_pass1(results, total_terms, terms_with_hits, terms_dict):
    """Generate markdown report for Pass 1 (grep)."""
    report = []
    report.append("# Chart Reference Extraction Report\n")
    report.append(f"**Chart**: `{CHART_FILE}`\n")
    report.append(f"**Textbook**: `{TEXTBOOK_MD}`\n")
    report.append(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    report.append("---\n\n")

    # Summary stats
    report.append("## Pass 1: Grep Search Results\n\n")
    report.append(f"- **Total searchable terms**: {total_terms}\n")
    report.append(f"- **Terms with ≥1 hit**: {terms_with_hits}\n")
    report.append(f"- **Terms with 0 hits**: {total_terms - terms_with_hits}\n")
    coverage = (terms_with_hits / total_terms * 100) if total_terms > 0 else 0
    report.append(f"- **Coverage**: {coverage:.1f}%\n\n")

    # Per-category breakdown
    report.append("### Per-Category Breakdown\n\n")
    report.append("| Category | Terms | With Hits | Coverage |\n")
    report.append("|----------|-------|-----------|----------|\n")
    for category, cat_results in results.items():
        cat_total = len(cat_results)
        cat_hits = sum(1 for r in cat_results if r["hit_count"] > 0)
        cat_cov = (cat_hits / cat_total * 100) if cat_total > 0 else 0
        report.append(f"| {category} | {cat_total} | {cat_hits} | {cat_cov:.0f}% |\n")
    report.append("\n")

    # Per-term details
    report.append("### Per-Term Results\n\n")
    for category, cat_results in results.items():
        report.append(f"#### {category}\n\n")
        for r in cat_results:
            status = f"✅ {r['hit_count']} hits" if r["hit_count"] > 0 else "❌ 0 hits"
            report.append(f"- **`{r['term']}`** ({r['note']}): {status}\n")
            if r["hit_count"] > 0 and r["hits"]:
                # Show first hit snippet
                first = r["hits"][0]
                snippet_preview = r["hits"][0]["snippet"][:200].replace("\n", " ")
                report.append(f"  - Line {first['line']}: _{snippet_preview}_\n")
        report.append("\n")

    # Gap list
    report.append("### Gap List (0 hits)\n\n")
    gap_count = 0
    for category, cat_results in results.items():
        gaps = [r for r in cat_results if r["hit_count"] == 0]
        if gaps:
            report.append(f"**{category}**:\n")
            for r in gaps:
                report.append(f"- `{r['term']}` ({r['note']})\n")
                gap_count += 1
            report.append("\n")
    report.append(f"**Total gaps: {gap_count}**\n\n")

    return "".join(report), gap_count


# ── Phase D: PageIndex RAG Pass ────────────────────────────────────────

def build_rag_queries(chart_data, gap_results):
    """Build natural-language queries for PageIndex RAG from gaps and chart context."""
    queries = []

    # Key chart-specific queries regardless of gaps
    planets = chart_data["planets"]
    lagna = chart_data["lagna"]
    karakas = chart_data["chara_karakas"]

    # Lagna-specific
    queries.append(("Libra ascendant characteristics and predictions", "Lagna"))

    # Atmakaraka
    if "AK" in karakas:
        ak_planet = karakas["AK"]["planet"]
        ak_info = planets.get(ak_planet, {})
        queries.append((
            f"Atmakaraka {ak_planet} in {RASI_FULL.get(ak_info.get('rasi',''), '')} significance",
            "Atmakaraka"
        ))

    # Exalted Jupiter
    if "Jupiter" in planets and planets["Jupiter"]["rasi"] == "Cn":
        queries.append(("Jupiter exalted in Cancer results for Libra lagna", "Exalted Jupiter"))

    # Stellium in Virgo (Sun, Mercury, Venus)
    virgo_planets = [p for p, info in planets.items() if info["rasi"] == "Vi"]
    if len(virgo_planets) >= 2:
        names = " and ".join(virgo_planets)
        queries.append((f"{names} conjunction in Virgo 12th house from Libra lagna", "Virgo stellium"))

    # Moon-Mars in Taurus (8th house from Libra)
    ta_planets = [p for p, info in planets.items() if info["rasi"] == "Ta"]
    if len(ta_planets) >= 2:
        names = " and ".join(ta_planets)
        queries.append((f"{names} in Taurus 8th house from Libra lagna", "Taurus combination"))

    # Saturn in Sagittarius (3rd house)
    if "Saturn" in planets and planets["Saturn"]["rasi"] == "Sg":
        queries.append(("Saturn in 3rd house Sagittarius from Libra lagna results", "Saturn 3rd"))

    # Rahu in Capricorn (4th house)
    if "Rahu" in planets and planets["Rahu"]["rasi"] == "Cp":
        queries.append(("Rahu in 4th house Capricorn results", "Rahu 4th"))

    # Ketu conjunct Jupiter in Cancer
    if "Ketu" in planets and planets["Ketu"]["rasi"] == "Cn":
        queries.append(("Ketu and Jupiter conjunction in Cancer 10th house", "Ketu-Jupiter"))

    # Dasa-specific (current period)
    queries.append(("Jupiter mahadasa Mars antardasa results", "Current dasa"))

    # Ashtakavarga significance
    queries.append(("how to interpret ashtakavarga sodhya pinda and transit predictions", "Ashtakavarga"))

    # Avasthas
    queries.append(("planetary avasthas bala kumara yuva mrita interpretation", "Avasthas"))

    # Navamsa analysis
    queries.append(("navamsa chart interpretation for marriage and dharma", "Navamsa"))

    # Dasamsa analysis
    queries.append(("dasamsa D-10 chart career profession interpretation", "Dasamsa"))

    return queries


# Keyword-to-chapter mapping for targeted RAG (supplements LLM tree search)
QUERY_CHAPTER_MAP = {
    "Lagna": ["C02", "C07", "C13"],
    "Atmakaraka": ["C08", "C13", "C09"],
    "Exalted Jupiter": ["C03", "C07", "C11", "C13"],
    "Virgo stellium": ["C07", "C10", "C11", "C13"],
    "Taurus combination": ["C07", "C10", "C13", "C14"],
    "Saturn 3rd": ["C07", "C10", "C15"],
    "Rahu 4th": ["C03", "C07", "C10"],
    "Ketu-Jupiter": ["C03", "C07", "C10", "C11"],
    "Current dasa": ["C16", "C17", "C13"],
    "Ashtakavarga": ["C12", "C25"],
    "Avasthas": ["C15", "C03"],
    "Navamsa": ["C06", "C13", "C07"],
    "Dasamsa": ["C06", "C13", "C07"],
}


def run_rag_pass(queries, chart_data):
    """Run PageIndex RAG queries and return results."""
    if not PAGEINDEX_AVAILABLE:
        return None, "PageIndex RAG not available (import failed)"

    # Check Ollama
    try:
        import requests as req
        resp = req.get(f"http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        model = RAG_MODEL if any(RAG_MODEL in m for m in models) else RAG_MODEL_FALLBACK
    except Exception:
        return None, "Ollama not reachable"

    # Monkey-patch the model in chat_pageindex
    import chat_pageindex
    chat_pageindex.MODEL = model

    print(f"  RAG model: {model}")
    reader = PdfReader(TEXTBOOK_PDF)

    rag_results = []
    for query, label in queries:
        print(f"  RAG query: [{label}] {query[:60]}...")
        try:
            # Step 1: Tree search (LLM) + keyword-based chapter mapping (deterministic)
            llm_result = tree_search(query, BOOK_TREE)
            llm_ids = set(llm_result.get("node_ids", [])) if llm_result else set()

            # Add keyword-mapped chapters
            mapped_ids = set(QUERY_CHAPTER_MAP.get(label, []))
            combined_ids = list(llm_ids | mapped_ids)

            if not combined_ids:
                rag_results.append({"query": query, "label": label, "status": "no_sections", "passages": []})
                continue

            print(f"    LLM nodes: {sorted(llm_ids)}, Mapped nodes: {sorted(mapped_ids)}, Combined: {sorted(combined_ids)}")

            # Step 2: Resolve nodes and extract pages
            nodes = resolve_nodes(BOOK_TREE, combined_ids)
            if not nodes:
                rag_results.append({"query": query, "label": label, "status": "no_nodes", "passages": []})
                continue

            all_pages = []
            for node in nodes:
                section_pages = extract_pages(reader, node["start_page"], node["end_page"])
                for page_num, text in section_pages:
                    score = score_page_relevance(query, text)
                    all_pages.append((page_num, text, score, node["title"]))

            all_pages.sort(key=lambda x: x[2], reverse=True)
            top_pages = all_pages[:6]  # take top 6 for broader coverage

            passages = []
            for pg, text, sc, sec in top_pages:
                passages.append({
                    "page": pg,
                    "section": sec,
                    "score": sc,
                    "text": text[:800],
                })

            rag_results.append({
                "query": query,
                "label": label,
                "status": "ok",
                "node_ids": sorted(combined_ids),
                "passages": passages,
            })

        except Exception as e:
            rag_results.append({"query": query, "label": label, "status": f"error: {e}", "passages": []})

    return rag_results, None


def generate_report_pass2(rag_results, error_msg=None):
    """Generate markdown for Pass 2 (RAG) results."""
    report = []
    report.append("---\n\n")
    report.append("## Pass 2: PageIndex RAG Results (Gap-Filling)\n\n")

    if error_msg:
        report.append(f"**Error**: {error_msg}\n\n")
        return "".join(report)

    if not rag_results:
        report.append("No RAG results generated.\n\n")
        return "".join(report)

    successful = sum(1 for r in rag_results if r["status"] == "ok" and r["passages"])
    report.append(f"- **Queries sent**: {len(rag_results)}\n")
    report.append(f"- **Successful retrievals**: {successful}\n")
    report.append(f"- **Failed/empty**: {len(rag_results) - successful}\n\n")

    for r in rag_results:
        report.append(f"### [{r['label']}] {r['query']}\n\n")
        if r["status"] != "ok":
            report.append(f"Status: {r['status']}\n\n")
            continue
        if not r["passages"]:
            report.append("No relevant passages found.\n\n")
            continue

        node_ids = r.get("node_ids", [])
        report.append(f"**Sections identified**: {', '.join(node_ids)}\n\n")
        for p in r["passages"]:
            report.append(f"**Page {p['page']}** ({p['section']}, relevance={p['score']})\n")
            # Show first 300 chars of text
            preview = p["text"][:300].replace("\n", " ")
            report.append(f"> {preview}...\n\n")

    return "".join(report)


# ── Phase E: Synonym/Broader Grep Re-run ───────────────────────────────

def generate_synonym_terms():
    """Additional terms using textbook's actual spellings and broader concepts."""
    extra = OrderedDict()

    extra["Sanskrit Sign Names"] = [
        ("Tula", "Libra in Sanskrit"),
        ("Kanya", "Virgo in Sanskrit"),
        ("Vrishabha", "Taurus in Sanskrit"),
        ("Karkataka", "Cancer in Sanskrit"),
        ("Dhanu", "Sagittarius in Sanskrit"),
        ("Makara", "Capricorn in Sanskrit"),
        ("Simha", "Leo in Sanskrit"),
        ("Mesha", "Aries in Sanskrit"),
        ("Mithuna", "Gemini in Sanskrit"),
        ("Vrischika", "Scorpio in Sanskrit"),
        ("Kumbha", "Aquarius in Sanskrit"),
        ("Meena", "Pisces in Sanskrit"),
    ]

    extra["Textbook Spelling Variants"] = [
        ("Amatya Karaka", "spaced form of Amatyakaraka"),
        ("Bhratri Karaka", "spaced form of Bhratrikaraka"),
        ("Matri Karaka", "spaced form of Matrikaraka"),
        ("Pitri Karaka", "spaced form of Pitrikaraka"),
        ("Putra Karaka", "spaced form of Putrakaraka"),
        ("Jnaati Karaka", "spaced form of Gnatikaraka"),
        ("Dara karaka", "spaced form of Darakaraka"),
        ("Atma Karaka", "spaced form of Atmakaraka"),
        ("Poorvashadha", "textbook spelling of Purva Ashadha"),
        ("Uttarashadha", "textbook spelling of Uttara Ashadha"),
        ("Bhinna Ashtakavarga", "textbook spelling of bhinnashtakavarga"),
        ("Sarva Ashtakavarga", "textbook spelling of sarvashtakavarga"),
        ("Bhrigu", "Bhrigu Bindu / Bhrigu transits"),
        ("Varnada", "Varnada Lagna concept"),
        ("Visaakha", "textbook spelling of Visakha"),
    ]

    extra["Broader Concepts"] = [
        ("benefic in dusthana", "benefic planet in 6/8/12"),
        ("malefic in kendra", "malefic in 1/4/7/10"),
        ("lord of 12th", "12th lord — Mercury for Libra"),
        ("lord of the 8th", "8th lord — Venus for Libra"),
        ("lord of the 10th", "10th lord — Moon for Libra"),
        ("9th lord", "dharma lord — Mercury for Libra"),
        ("lagna lord", "lord of ascendant"),
        ("yogakaraka", "planet giving raja yoga"),
        ("badhaka", "obstructing planet/sign"),
        ("mahadasa lord", "ruler of major period"),
        ("kendradhipati", "lord of angular house"),
        ("trikonadhipati", "lord of trinal house"),
        ("dharmakarmadhipati", "9th-10th lord conjunction"),
    ]

    extra["Specific Combinations"] = [
        ("Mercury and Venus", "conjunction in Virgo"),
        ("Sun and Mercury", "Budhaditya in 12th"),
        ("Moon and Mars", "combination in Taurus/8th"),
        ("Jupiter and Ketu", "conjunction in Cancer/10th"),
        ("Saturn aspects", "Saturn's special aspects 3rd 10th"),
        ("Rahu in Capricorn", "Rahu in 4th house"),
        ("Moon as Atmakaraka", "AK Moon significance"),
    ]

    return extra


# ── Main Pipeline ──────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Chart-to-Textbook Reference Extraction Pipeline")
    print("  Hybrid: Grep + PageIndex RAG")
    print("=" * 70)

    # Phase A: Parse chart
    print("\n[Phase A] Parsing chart...")
    chart_data = parse_chart(CHART_FILE)
    print(f"  Planets parsed: {list(chart_data['planets'].keys())}")
    print(f"  Lagna: {chart_data['lagna'].get('rasi', '?')} ({RASI_FULL.get(chart_data['lagna'].get('rasi',''), '?')})")
    print(f"  Karakas: {list(chart_data['chara_karakas'].keys())}")
    print(f"  Special lagnas: {len(chart_data['special_lagnas'])}")
    print(f"  Avasthas for: {list(chart_data['avasthas'].keys())}")

    print("\n[Phase A] Generating searchable terms...")
    terms = generate_terms(chart_data)
    total_count = sum(len(v) for v in terms.values())
    print(f"  Generated {total_count} terms across {len(terms)} categories")
    for cat, tlist in terms.items():
        print(f"    {cat}: {len(tlist)} terms")

    # Phase B: Grep pass
    print("\n[Phase B] Loading textbook...")
    lines = load_textbook(TEXTBOOK_MD)
    print(f"  Loaded {len(lines)} lines from {TEXTBOOK_MD}")

    print("\n[Phase B] Running grep pass (Pass 1)...")
    results, total_terms, terms_with_hits = run_grep_pass(terms, lines)
    print(f"  Total terms: {total_terms}")
    print(f"  Terms with hits: {terms_with_hits}")
    print(f"  Coverage: {terms_with_hits/total_terms*100:.1f}%")

    # Phase C: Report Pass 1
    print("\n[Phase C] Generating Pass 1 report...")
    report_pass1, gap_count = generate_report_pass1(results, total_terms, terms_with_hits, terms)
    print(f"  Gaps identified: {gap_count}")

    # Phase D: Synonym grep re-run
    print("\n[Phase D-1] Running synonym/broader grep pass...")
    extra_terms = generate_synonym_terms()
    extra_count = sum(len(v) for v in extra_terms.values())
    print(f"  Extra terms: {extra_count}")
    extra_results, extra_total, extra_hits = run_grep_pass(extra_terms, lines)
    print(f"  Extra terms with hits: {extra_hits}/{extra_total}")

    # Append extra results to report
    report_extra = []
    report_extra.append("---\n\n")
    report_extra.append("## Pass 1.5: Synonym & Broader Term Grep\n\n")
    report_extra.append(f"- **Additional terms**: {extra_total}\n")
    report_extra.append(f"- **Terms with hits**: {extra_hits}\n")
    extra_cov = (extra_hits / extra_total * 100) if extra_total > 0 else 0
    report_extra.append(f"- **Coverage**: {extra_cov:.1f}%\n\n")
    for category, cat_results in extra_results.items():
        report_extra.append(f"### {category}\n\n")
        for r in cat_results:
            status = f"✅ {r['hit_count']} hits" if r["hit_count"] > 0 else "❌ 0 hits"
            report_extra.append(f"- **`{r['term']}`** ({r['note']}): {status}\n")
            if r["hit_count"] > 0 and r["hits"]:
                first = r["hits"][0]
                snippet_preview = first["snippet"][:200].replace("\n", " ")
                report_extra.append(f"  - Line {first['line']}: _{snippet_preview}_\n")
        report_extra.append("\n")

    # Phase D-2: PageIndex RAG pass
    print("\n[Phase D-2] Building RAG queries...")
    rag_queries = build_rag_queries(chart_data, results)
    print(f"  RAG queries: {len(rag_queries)}")

    print("\n[Phase D-2] Running PageIndex RAG pass (Pass 2)...")
    rag_results, rag_error = run_rag_pass(rag_queries, chart_data)

    # Phase E: Report Pass 2
    print("\n[Phase E] Generating Pass 2 report...")
    report_pass2 = generate_report_pass2(rag_results, rag_error)

    # Final summary
    report_final = []
    report_final.append("---\n\n")
    report_final.append("## Final Summary\n\n")

    combined_terms = total_terms + extra_total
    combined_hits = terms_with_hits + extra_hits
    combined_cov = (combined_hits / combined_terms * 100) if combined_terms > 0 else 0
    report_final.append(f"- **Pass 1 grep terms**: {total_terms} ({terms_with_hits} with hits, {terms_with_hits/total_terms*100:.1f}%)\n")
    report_final.append(f"- **Pass 1.5 synonym terms**: {extra_total} ({extra_hits} with hits, {extra_cov:.1f}%)\n")
    report_final.append(f"- **Combined grep coverage**: {combined_hits}/{combined_terms} = {combined_cov:.1f}%\n")

    if rag_results:
        rag_ok = sum(1 for r in rag_results if r["status"] == "ok" and r["passages"])
        report_final.append(f"- **RAG queries**: {len(rag_results)} ({rag_ok} successful)\n")
    else:
        report_final.append(f"- **RAG queries**: N/A ({rag_error})\n")

    report_final.append(f"\n### Remaining Gaps\n\n")
    # Collect all zero-hit terms from both passes
    all_gaps = []
    for category, cat_results in results.items():
        for r in cat_results:
            if r["hit_count"] == 0:
                all_gaps.append((category, r["term"], r["note"]))
    # Remove those covered by extra pass
    extra_covered = set()
    for category, cat_results in extra_results.items():
        for r in cat_results:
            if r["hit_count"] > 0:
                extra_covered.add(r["term"].lower())

    remaining = [(c, t, n) for c, t, n in all_gaps if t.lower() not in extra_covered]
    report_final.append(f"After both grep passes, **{len(remaining)}** original terms still have 0 hits.\n\n")
    if remaining:
        for c, t, n in remaining[:30]:  # cap display
            report_final.append(f"- [{c}] `{t}` ({n})\n")
        if len(remaining) > 30:
            report_final.append(f"- ...and {len(remaining) - 30} more\n")

    # Write full report
    full_report = report_pass1 + "".join(report_extra) + report_pass2 + "".join(report_final)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(full_report)
    print(f"\n{'='*70}")
    print(f"  Report saved to: {REPORT_FILE}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
