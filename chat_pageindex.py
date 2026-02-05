"""
PageIndex-style Reasoning-Based RAG for Jyotish Holocron
=========================================================
Instead of vector similarity search, this uses:
1. A hierarchical tree index of the document (like a smart TOC)
2. LLM reasoning to navigate the tree and find relevant sections
3. Direct page extraction from those sections

Inspired by https://github.com/VectifyAI/PageIndex
"""

import json
import sys
import re
import requests
from pypdf import PdfReader

PDF_PATH = "vedic_astro_textbook.pdf"
MODEL = "qwen2.5:3b"
OLLAMA_URL = "http://localhost:11434"
# Book page 1 = PDF page 13 (12 pages of front matter before page numbering starts)
PDF_PAGE_OFFSET = 12

# ── Hierarchical Tree Index (PageIndex-style) ──────────────────────────
# This tree mirrors the book's actual structure. Each node has:
#   title, summary, start_page, end_page, and optional child nodes.
# The LLM reasons over this tree to find relevant sections.

BOOK_TREE = {
    "title": "Vedic Astrology: An Integrated Approach by P.V.R. Narasimha Rao",
    "total_pages": 512,
    "nodes": [
        {
            "node_id": "P1",
            "title": "Part 1: Chart Analysis",
            "summary": "Fundamental principles and techniques of chart analysis. Covers signs, planets, houses, divisional charts, aspects, yogas, and chart interpretation.",
            "start_page": 1,
            "end_page": 206,
            "nodes": [
                {"node_id": "C01", "title": "1. Basic Concepts", "summary": "Introduction to Vedic astrology, zodiac, planets, houses, nakshatras, basic computational framework.", "start_page": 3, "end_page": 20},
                {"node_id": "C02", "title": "2. Rasis (Signs)", "summary": "Characteristics and properties of 12 zodiac signs (Aries-Pisces), movable/fixed/dual classification, elements.", "start_page": 21, "end_page": 27},
                {"node_id": "C03", "title": "3. Planets", "summary": "Nature, significations and characteristics of Sun, Moon, Mars, Mercury, Jupiter, Venus, Saturn, Rahu, Ketu. Exaltation, debilitation, ownership.", "start_page": 28, "end_page": 40},
                {"node_id": "C04", "title": "4. Upagrahas", "summary": "Sub-planets: Dhuma, Vyatipata, Parivesha, Indrachapa, Upaketu and their computation.", "start_page": 41, "end_page": 44},
                {"node_id": "C05", "title": "5. Special Lagnas", "summary": "Hora lagna, ghati lagna, sree lagna, varnada lagna and other special ascendants.", "start_page": 45, "end_page": 50},
                {"node_id": "C06", "title": "6. Divisional Charts", "summary": "Vargas D-1 through D-60. Rasi chart, navamsa, dasamamsa, etc. How to compute and use divisional charts.", "start_page": 51, "end_page": 66},
                {"node_id": "C07", "title": "7. Houses", "summary": "Significations of 12 houses (bhavas), house lords, functional nature of planets for each lagna, kendras, trikonas, dusthanas.", "start_page": 67, "end_page": 78},
                {"node_id": "C08", "title": "8. Karakas", "summary": "Chara karakas (variable significators), sthira karakas (fixed significators), naisargika karakas (natural significators).", "start_page": 79, "end_page": 84},
                {"node_id": "C09", "title": "9. Arudha Padas", "summary": "Computation and interpretation of arudha padas (AL, A2-A12), upapada lagna, use of arudha padas in prediction.", "start_page": 85, "end_page": 99},
                {"node_id": "C10", "title": "10. Aspects and Argalas", "summary": "Graha drishti (planetary aspects) - all planets aspect 7th house; special aspects of Mars (4th,8th), Jupiter (5th,9th), Saturn (3rd,10th). Rasi drishti (sign aspects). Argalas (planetary interventions) and virodha argalas.", "start_page": 100, "end_page": 111},
                {"node_id": "C11", "title": "11. Yogas", "summary": "Important planetary combinations: raja yogas, dhana yogas, aristha yogas, pancha mahapurusha yogas, Gajakesari, Budhaditya, and many more.", "start_page": 112, "end_page": 144},
                {"node_id": "C12", "title": "12. Ashtakavarga", "summary": "Ashtakavarga system: bhinnashtakavarga, sarvashtakavarga, kaksha, transit predictions using ashtakavarga.", "start_page": 145, "end_page": 165},
                {"node_id": "C13", "title": "13. Interpreting Charts", "summary": "Systematic approach to chart interpretation, analyzing houses, planets, dasas together.", "start_page": 166, "end_page": 179},
                {"node_id": "C14", "title": "14. Topics Related to Longevity", "summary": "Determining longevity, marakas (death-inflicting planets), timing of death.", "start_page": 180, "end_page": 186},
                {"node_id": "C15", "title": "15. Strength of Planets and Rasis", "summary": "Methods to compare strengths: shadbala, rules for comparing planet and sign strengths.", "start_page": 187, "end_page": 206}
            ]
        },
        {
            "node_id": "P2",
            "title": "Part 2: Dasa Analysis",
            "summary": "Timing events using planetary period (dasa) systems. Covers multiple dasa systems for different purposes.",
            "start_page": 207,
            "end_page": 312,
            "nodes": [
                {"node_id": "C16", "title": "16. Vimsottari Dasa", "summary": "120-year nakshatra-based dasa system, most commonly used. Mahadasa, antardasa, pratyantardasa computation and interpretation.", "start_page": 209, "end_page": 225},
                {"node_id": "C17", "title": "17. Ashtottari Dasa", "summary": "108-year conditional dasa system.", "start_page": 226, "end_page": 230},
                {"node_id": "C18", "title": "18. Narayana Dasa", "summary": "Rasi (sign) based dasa system taught by Parasara. Computation and interpretation.", "start_page": 231, "end_page": 258},
                {"node_id": "C19", "title": "19. Lagna Kendradi Rasi Dasa", "summary": "Sign-based dasa starting from lagna for specific purposes.", "start_page": 259, "end_page": 262},
                {"node_id": "C20", "title": "20. Sudasa", "summary": "Dasa for wealth and prosperity analysis using sree lagna.", "start_page": 263, "end_page": 266},
                {"node_id": "C21", "title": "21. Drigdasa", "summary": "Aspect-based sign dasa system.", "start_page": 267, "end_page": 272},
                {"node_id": "C22", "title": "22. Niryaana Shoola Dasa", "summary": "Dasa specifically for timing death.", "start_page": 273, "end_page": 280},
                {"node_id": "C23", "title": "23. Shoola Dasa", "summary": "Dasa for timing suffering and difficult periods.", "start_page": 281, "end_page": 288},
                {"node_id": "C24", "title": "24. Kalachakra Dasa", "summary": "Complex nakshatra-based dasa using the kalachakra (wheel of time).", "start_page": 289, "end_page": 312}
            ]
        },
        {
            "node_id": "P3",
            "title": "Part 3: Transit Analysis",
            "summary": "Analyzing effects of planetary transits over natal chart positions.",
            "start_page": 313,
            "end_page": 364,
            "nodes": [
                {"node_id": "C25", "title": "25. Transits and Natal References", "summary": "How transiting planets interact with natal positions, Sade Sati (Saturn transit over Moon), transit rules.", "start_page": 314, "end_page": 345},
                {"node_id": "C26", "title": "26. Transits: Miscellaneous Topics", "summary": "Vedha, sarvatobhadra chakra, transit-based predictions.", "start_page": 346, "end_page": 364}
            ]
        },
        {
            "node_id": "P4",
            "title": "Part 4: Tajaka Analysis",
            "summary": "Annual chart (varshaphal/solar return) analysis techniques.",
            "start_page": 365,
            "end_page": 416,
            "nodes": [
                {"node_id": "C27", "title": "27. Tajaka Chart Basics", "summary": "Fundamentals of annual/solar return charts.", "start_page": 367, "end_page": 373},
                {"node_id": "C28", "title": "28. Techniques of Tajaka Charts", "summary": "Sahams, muntha, harsha bala, pancha vargeeya bala in Tajaka.", "start_page": 374, "end_page": 385},
                {"node_id": "C29", "title": "29. Tajaka Yogas", "summary": "Itthasala, Ishrafa, Nakta, Yamaya, and other Tajaka yogas.", "start_page": 386, "end_page": 395},
                {"node_id": "C30", "title": "30. Annual Dasas", "summary": "Mudda dasa, yogini dasa for annual charts.", "start_page": 396, "end_page": 407},
                {"node_id": "C31", "title": "31. Sudarsana Chakra Dasa", "summary": "Three-reference dasa system using lagna, Sun and Moon.", "start_page": 408, "end_page": 416}
            ]
        },
        {
            "node_id": "P5",
            "title": "Part 5: Special Topics",
            "summary": "Advanced and miscellaneous topics: birthtime rectification, remedies, mundane astrology, muhurta.",
            "start_page": 417,
            "end_page": 484,
            "nodes": [
                {"node_id": "C32", "title": "32. Impact of Birthtime Error", "summary": "How birthtime inaccuracies affect divisional charts and predictions.", "start_page": 420, "end_page": 440},
                {"node_id": "C33", "title": "33. Rational Thinking", "summary": "Scientific and rational approach to Vedic astrology.", "start_page": 441, "end_page": 449},
                {"node_id": "C34", "title": "34. Remedial Measures", "summary": "Mantras, gemstones, charity and other astrological remedies.", "start_page": 450, "end_page": 459},
                {"node_id": "C35", "title": "35. Mundane Astrology", "summary": "Astrology of nations, world events, natural disasters.", "start_page": 460, "end_page": 470},
                {"node_id": "C36", "title": "36. Muhurta or Electional Astrology", "summary": "Choosing auspicious times (muhurtas) for important activities.", "start_page": 471, "end_page": 481},
                {"node_id": "C37", "title": "37. Ethical Behavior of a Jyotishi", "summary": "Rules and ethics for practicing Vedic astrologers.", "start_page": 482, "end_page": 484}
            ]
        },
        {
            "node_id": "P6",
            "title": "Part 6: Real-life Examples",
            "summary": "Practical chart interpretations and real-life example horoscopes demonstrating techniques from the book.",
            "start_page": 485,
            "end_page": 512,
            "nodes": []
        }
    ]
}


# ── Core Functions ──────────────────────────────────────────────────────

def extract_pages(reader, start_page, end_page):
    """Extract text from specific book pages of the PDF. Returns list of (book_page, text).
    Applies PDF_PAGE_OFFSET to convert book page numbers to PDF page indices."""
    pages = []
    pdf_start = start_page + PDF_PAGE_OFFSET - 1  # 0-indexed PDF page
    pdf_end = end_page + PDF_PAGE_OFFSET            # exclusive upper bound
    for i in range(pdf_start, min(pdf_end, len(reader.pages))):
        page_text = (reader.pages[i].extract_text() or "").strip()
        if page_text:
            book_page = i - PDF_PAGE_OFFSET + 1
            pages.append((book_page, page_text))
    return pages


def score_page_relevance(query, page_text):
    """Score a page's relevance to the query using keyword matching."""
    query_lower = query.lower()
    text_lower = page_text.lower()
    keywords = [w.strip() for w in query_lower.split() if len(w.strip()) > 2]
    score = 0
    for kw in keywords:
        count = text_lower.count(kw)
        score += count
    # Bonus for exact phrase match
    if query_lower in text_lower:
        score += 20
    # Bonus for partial phrase matches (bigrams)
    words = keywords
    for i in range(len(words) - 1):
        bigram = f"{words[i]} {words[i+1]}"
        if bigram in text_lower:
            score += 5
    return score


def call_ollama(prompt, model=MODEL, num_ctx=4096):
    """Call Ollama API and return the response text."""
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0, "num_ctx": num_ctx},
        },
        timeout=300,
    )
    if resp.status_code != 200:
        print(f"  Ollama error: {resp.text[:300]}")
        resp.raise_for_status()
    return resp.json()["response"]


def build_tree_for_prompt(tree):
    """Build a compact tree representation for the LLM prompt."""
    lines = [f"Document: {tree['title']} ({tree['total_pages']} pages)\n"]
    for part in tree["nodes"]:
        lines.append(f"  [{part['node_id']}] {part['title']} (pp. {part['start_page']}-{part['end_page']})")
        lines.append(f"      Summary: {part['summary']}")
        for ch in part.get("nodes", []):
            lines.append(f"    [{ch['node_id']}] {ch['title']} (pp. {ch['start_page']}-{ch['end_page']})")
            lines.append(f"        Summary: {ch['summary']}")
    return "\n".join(lines)


def tree_search(query, tree):
    """Use LLM reasoning to identify relevant sections from the tree index."""
    tree_text = build_tree_for_prompt(tree)

    prompt = f"""Below is the table of contents of a Vedic astrology textbook. Each section has an ID, title, page range, and summary.

{tree_text}

Question: {query}

Task: Read EVERY section summary carefully. List ALL section IDs (like C01, C10, etc.) whose title or summary is related to the question. A section is relevant if its summary mentions any keyword from the question.

Output format - return ONLY a JSON object like this:
{{"node_ids": ["C10", "C03"]}}

Do not explain. Return only the JSON."""

    response = call_ollama(prompt, num_ctx=4096)
    print(f"  Raw LLM response: {response[:300]}")

    # Parse JSON from response
    try:
        match = re.search(r'\{[^}]*"node_ids"\s*:\s*\[[^\]]*\][^}]*\}', response, re.DOTALL)
        if match:
            return json.loads(match.group())
    except json.JSONDecodeError:
        pass

    # Fallback: try to extract node IDs directly
    ids = re.findall(r'C\d{2}|P\d', response)
    if ids:
        return {"node_ids": list(set(ids))}

    return None


def resolve_nodes(tree, node_ids):
    """Resolve node_ids to their page ranges."""
    node_map = {}
    for part in tree["nodes"]:
        node_map[part["node_id"]] = part
        for ch in part.get("nodes", []):
            node_map[ch["node_id"]] = ch

    resolved = []
    for nid in node_ids:
        if nid in node_map:
            resolved.append(node_map[nid])
        else:
            # Try fuzzy match
            for key in node_map:
                if key.lower() == nid.lower():
                    resolved.append(node_map[key])
                    break
    return resolved


def generate_answer(query, context):
    """Generate answer from retrieved context using LLM."""
    prompt = f"""You are a Jyotish (Vedic Astrology) expert assistant.
Answer the question using ONLY the provided book excerpts below.
If the answer is not in the excerpts, say "Not found in the book."
Always cite page numbers from the excerpts.
Be thorough and detailed in your answer.

Book Excerpts:
{context}

Question: {query}

Answer:"""

    return call_ollama(prompt, num_ctx=8192)


# ── Main ────────────────────────────────────────────────────────────────

def query_once(reader, query):
    """Run a single query through the PageIndex-style RAG pipeline."""

    # Step 1: Tree Search — LLM reasons over the TOC to find sections
    print("  [Step 1] Reasoning over document tree index...")
    result = tree_search(query, BOOK_TREE)

    if not result or not result.get("node_ids"):
        print("  Could not identify relevant sections.")
        return None

    print(f"  Identified nodes: {result['node_ids']}")

    # Step 2: Resolve nodes, extract all pages, score by keyword relevance
    nodes = resolve_nodes(BOOK_TREE, result["node_ids"])
    if not nodes:
        print("  Could not resolve any nodes.")
        return None

    print(f"  [Step 2] Extracting & scoring pages from {len(nodes)} section(s)...")
    all_pages = []  # list of (page_num, text, score, section_title)

    for node in nodes:
        section_pages = extract_pages(reader, node["start_page"], node["end_page"])
        for page_num, text in section_pages:
            score = score_page_relevance(query, text)
            all_pages.append((page_num, text, score, node["title"]))

    # Sort by relevance score (descending), take top pages
    all_pages.sort(key=lambda x: x[2], reverse=True)
    top_n = 4  # Send only the top N most relevant pages
    selected = all_pages[:top_n]

    print(f"  Scored {len(all_pages)} pages. Top {top_n} by relevance:")
    for pg, _, sc, sec in selected:
        print(f"    Page {pg} (score={sc}) from '{sec}'")

    # Build context from selected pages only
    context_parts = []
    for pg, text, _, sec in sorted(selected, key=lambda x: x[0]):  # sort by page order
        context_parts.append(f"[Page {pg} — {sec}]\n{text}")

    context = "\n\n".join(context_parts)
    print(f"  Context size: {len(context)} chars")

    # Step 3: Generate answer from focused context
    print("  [Step 3] Generating answer...")
    answer = generate_answer(query, context)
    return answer


def main():
    print("=" * 60)
    print("Jyotish Holocron - PageIndex-style Reasoning RAG")
    print("=" * 60)
    print("Loading PDF...")
    reader = PdfReader(PDF_PATH)
    print(f"PDF loaded: {len(reader.pages)} pages")
    print(f"Model: {MODEL}")
    print(f"Tree index: {sum(len(p.get('nodes', [])) for p in BOOK_TREE['nodes'])} sections in {len(BOOK_TREE['nodes'])} parts")
    print("-" * 60)

    # If a query is passed as argument, run it and exit
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"\nQ> {query}\n")
        answer = query_once(reader, query)
        if answer:
            print(f"\nA> {answer}\n")
        return

    # Interactive mode
    print("Type 'exit' to quit.\n")
    while True:
        try:
            q = input("Q> ").strip()
            if q.lower() in {"exit", "quit"}:
                break
            if not q:
                continue

            answer = query_once(reader, q)
            if answer:
                print(f"\nA> {answer}\n")
            else:
                print("\nA> Could not find relevant information.\n")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\n  Error: {e}\n")
            continue


if __name__ == "__main__":
    main()
