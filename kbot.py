import asyncio
import json
import os
import random
import re
import time
from groq import AsyncGroq
from playwright.async_api import async_playwright, Page

# Load environment variables from GitHub Actions
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
KAHOOT_PIN = os.environ.get("KAHOOT_PIN", "").strip()

FIRST_CLASS_COUNT = min(max(int(os.environ.get("FIRST_CLASS_COUNT", "1")), 1), 2)
FIRST_CLASS_NAME_1 = os.environ.get("FIRST_CLASS_NAME_1", "Bot1").strip() or "Bot1"
RAW_FIRST_CLASS_NAME_2 = os.environ.get("FIRST_CLASS_NAME_2", "").strip()

SECOND_CLASS_COUNT = max(int(os.environ.get("SECOND_CLASS_COUNT", "0")), 0)
SECOND_CLASS_BASE_NAME = os.environ.get("SECOND_CLASS_NAME", "Ref").strip() or "Ref"

MODEL_NAME = "llama-3.3-70b-versatile"

if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY environment variable. Set it in GitHub Secrets.")

groq_client = AsyncGroq(api_key=GROQ_API_KEY)


def build_bot_roster():
    """Builds the list of 1st-class and 2nd-class bots based on workflow inputs."""
    roster = [{"name": FIRST_CLASS_NAME_1, "tier": 1}]

    if FIRST_CLASS_COUNT == 2:
        name_2 = RAW_FIRST_CLASS_NAME_2 if RAW_FIRST_CLASS_NAME_2 else "[default]"
        roster.append({"name": name_2, "tier": 1})

    for i in range(1, SECOND_CLASS_COUNT + 1):
        bot_name = f"{SECOND_CLASS_BASE_NAME}{i}" if SECOND_CLASS_COUNT > 1 else SECOND_CLASS_BASE_NAME
        roster.append({"name": bot_name, "tier": 2})

    return roster


# ---------------------------------------------------------------------------
# Ultra-Compact Groq Solvers (< 160 total tokens per call)
# ---------------------------------------------------------------------------

async def solve_mcq(question: str, options: list[str]) -> int:
    """Quiz / True-False / Poll: returns single best index."""
    opts = "\n".join(f"{i}:{o[:60]}" for i, o in enumerate(options))
    try:
        res = await groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Reply ONLY with the correct option digit (0-3)."},
                {"role": "user", "content": f"Q: {question[:180]}\n{opts}"},
            ],
            temperature=0.0,
            max_tokens=3,
        )
        m = re.search(r"[0-3]", res.choices[0].message.content)
        if m and int(m.group(0)) < len(options):
            return int(m.group(0))
    except Exception as e:
        print(f"[Groq MCQ Error] {e}")
    return 0


async def solve_multiselect(question: str, options: list[str]) -> list[int]:
    """Multi-Select: returns list of correct indices."""
    opts = "\n".join(f"{i}:{o[:60]}" for i, o in enumerate(options))
    try:
        res = await groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": 'JSON only: {"c":[indices]}'},
                {"role": "user", "content": f"Q: {question[:180]}\n{opts}"},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=25,
        )
        data = json.loads(res.choices[0].message.content)
        valid = sorted({int(i) for i in data.get("c", [0]) if 0 <= int(i) < len(options)})
        return valid if valid else [0]
    except Exception as e:
        print(f"[Groq Multi Error] {e}")
    return [0]


async def solve_puzzle(question: str, items: list[str]) -> list[int]:
    """Puzzle / Jumble: returns ordered permutation of indices (e.g. [2,0,3,1])."""
    opts = "\n".join(f"{i}:{o[:60]}" for i, o in enumerate(items))
    n = len(items)
    try:
        res = await groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": 'Order indices correctly. JSON only: {"o":[...]}'},
                {"role": "user", "content": f"Q: {question[:180]}\n{opts}"},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=30,
        )
        data = json.loads(res.choices[0].message.content)
        order = [int(i) for i in data.get("o", []) if 0 <= int(i) < n]
        if len(order) == n and len(set(order)) == n:
            return order
    except Exception as e:
        print(f"[Groq Puzzle Error] {e}")
    return list(range(n))


async def solve_text(question: str) -> dict:
    """Type Answer / Word Cloud / Open-Ended: returns 1 best + 3 wrong strings."""
    try:
        res = await groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": 'JSON only: {"b":"best (<20 chars)","w":["wrong1","wrong2","wrong3"]}',
                },
                {"role": "user", "content": f"Q: {question[:180]}"},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=60,
        )
        data = json.loads(res.choices[0].message.content)
        best = str(data.get("b", "Yes"))[:20].strip()
        wrong = [str(w)[:20].strip() for w in data.get("w", ["No", "Maybe", "None"])[:3]]
        while len(wrong) < 3:
            wrong.append("Unknown")
        return {"best": best, "wrong": wrong}
    except Exception as e:
        print(f"[Groq Text Error] {e}")
    return {"best": "Yes", "wrong": ["No", "Maybe", "Unknown"]}


async def solve_slider(question: str, min_v: float, max_v: float, step: float) -> dict:
    """Slider: queries Groq ONLY for the single best number; computes 3 wrong offsets locally."""
    def snap(val: float) -> float:
        clamped = max(min_v, min(max_v, float(val)))
        if step > 0:
            clamped = min_v + round((clamped - min_v) / step) * step
        return round(clamped, 4)

    best = snap((min_v + max_v) / 2)
    try:
        res = await groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": 'JSON only: {"v":number}'},
                {"role": "user", "content": f"Q: {question[:180]}\nRange:[{min_v},{max_v}]"},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=20,
        )
        data = json.loads(res.choices[0].message.content)
        best = snap(data.get("v", best))
    except Exception as e:
        print(f"[Groq Slider Error] {e}")

    # Generate 3 distinct wrong values locally to save tokens
    span = max_v - min_v
    offsets = [-0.25 * span, 0.2 * span, 0.4 * span]
    wrong = []
    for off in offsets:
        cand = snap(best + off)
        if cand == best:
            cand = snap(min_v if best > min_v else max_v)
        wrong.append(cand)
    return {"best": best, "wrong": wrong}


async def solve_pin(question: str) -> tuple[float, float]:
    """Pin on Image / Drop Pin: returns normalized (x_pct, y_pct) in [0.15, 0.85]."""
    try:
        res = await groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": 'Estimate target image coordinates (0.1 to 0.9). JSON only: {"x":0.5,"y":0.5}',
                },
                {"role": "user", "content": f"Q: {question[:180]}"},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=25,
        )
        data = json.loads(res.choices[0].message.content)
        x = max(0.1, min(0.9, float(data.get("x", 0.5))))
        y = max(0.1, min(0.9, float(data.get("y", 0.5))))
        return (x, y)
    except Exception:
        return (0.5, 0.5)


# ---------------------------------------------------------------------------
# Tier Selection Logic (Tier 1 = 100% Best | Tier 2 = 50% Best, 50% of 3 Wrong)
# ---------------------------------------------------------------------------

def pick_mcq(tier: int, best_idx: int, n: int) -> int:
    if tier == 1 or n <= 1 or random.random() < 0.5:
        return best_idx
    wrong = [i for i in range(n) if i != best_idx]
    return random.choice(wrong)


def pick_multiselect(tier: int, best_indices: list[int], n: int) -> list[int]:
    if tier == 1 or n <= 1 or random.random() < 0.5:
        return best_indices
    all_idx = list(range(n))
    wrong_subsets = []
    for _ in range(15):
        cand = sorted(random.sample(all_idx, random.randint(1, max(1, n - 1))))
        if cand != best_indices and cand not in wrong_subsets:
            wrong_subsets.append(cand)
        if len(wrong_subsets) == 3:
            break
    return random.choice(wrong_subsets) if wrong_subsets else [0]


def pick_puzzle(tier: int, best_order: list[int]) -> list[int]:
    if tier == 1 or random.random() < 0.5:
        return best_order
    wrong_orders = []
    for _ in range(15):
        cand = best_order[:]
        random.shuffle(cand)
        if cand != best_order and cand not in wrong_orders:
            wrong_orders.append(cand)
        if len(wrong_orders) == 3:
            break
    return random.choice(wrong_orders) if wrong_orders else best_order[::-1]


# ---------------------------------------------------------------------------
# Bot Action Dispatchers (1.5 - 3.0s Window)
# ---------------------------------------------------------------------------

async def wait_window(reveal_timestamp: float):
    target_delay = random.uniform(1.5, 3.0)
    await asyncio.sleep(max(0.0, target_delay - (time.monotonic() - reveal_timestamp)))


async def click_submit_if_present(page: Page):
    submit_sel = (
        'button[data-functional-selector*="submit"], '
        'button[type="submit"]'
    )
    btn = page.locator(submit_sel).first
    if await btn.count() > 0 and await btn.is_visible(timeout=800):
        await btn.click()


async def submit_mcq_bot(bot: dict, best_idx: int, n: int, reveal_ts: float):
    await wait_window(reveal_ts)
    idx = pick_mcq(bot["tier"], best_idx, n)
    try:
        btn = bot["page"].locator(f'[data-functional-selector="answer-{idx}"]')
        if await btn.is_visible(timeout=1000):
            await btn.click()
            print(f"[MCQ] {bot['name']} (C{bot['tier']}) -> {idx} ({time.monotonic() - reveal_ts:.2f}s)")
    except Exception as e:
        print(f"[MCQ Err] {bot['name']}: {e}")


async def submit_multi_bot(bot: dict, best_indices: list[int], n: int, reveal_ts: float):
    await wait_window(reveal_ts)
    indices = pick_multiselect(bot["tier"], best_indices, n)
    try:
        for idx in indices:
            btn = bot["page"].locator(f'[data-functional-selector="answer-{idx}"]')
            if await btn.is_visible(timeout=800):
                await btn.click()
                await asyncio.sleep(0.06)
        await click_submit_if_present(bot["page"])
        print(f"[Multi] {bot['name']} (C{bot['tier']}) -> {indices} ({time.monotonic() - reveal_ts:.2f}s)")
    except Exception as e:
        print(f"[Multi Err] {bot['name']}: {e}")


async def submit_puzzle_bot(bot: dict, items: list[str], best_order: list[int], reveal_ts: float):
    await wait_window(reveal_ts)
    target_order = pick_puzzle(bot["tier"], best_order)
    page: Page = bot["page"]

    card_sel = '[data-functional-selector*="jumble-card"], [data-functional-selector*="puzzle-card"], [draggable="true"]'
    try:
        for target_pos, orig_idx in enumerate(target_order):
            target_text = items[orig_idx]
            cards = page.locator(card_sel)
            count = await cards.count()
            # Find where target_text currently sits and drag it to target_pos
            for cur_pos in range(count):
                txt = (await cards.nth(cur_pos).inner_text()).strip()
                if txt == target_text and cur_pos != target_pos:
                    src_box = await cards.nth(cur_pos).bounding_box()
                    dst_box = await cards.nth(target_pos).bounding_box()
                    if src_box and dst_box:
                        await page.mouse.move(src_box["x"] + src_box["width"] / 2, src_box["y"] + src_box["height"] / 2)
                        await page.mouse.down()
                        await page.mouse.move(
                            dst_box["x"] + dst_box["width"] / 2, dst_box["y"] + dst_box["height"] / 2, steps=6
                        )
                        await page.mouse.up()
                        await asyncio.sleep(0.08)
                    break
        await click_submit_if_present(page)
        print(f"[Puzzle] {bot['name']} (C{bot['tier']}) -> {target_order} ({time.monotonic() - reveal_ts:.2f}s)")
    except Exception as e:
        print(f"[Puzzle Err] {bot['name']}: {e}")


async def submit_text_bot(bot: dict, text_pack: dict, reveal_ts: float):
    await wait_window(reveal_ts)
    ans = text_pack["best"] if (bot["tier"] == 1 or random.random() < 0.5) else random.choice(text_pack["wrong"])
    page: Page = bot["page"]
    input_sel = (
        'input[data-functional-selector*="text-answer"], '
        'textarea[data-functional-selector*="open-ended"], '
        'input[placeholder*="answer" i], textarea'
    )
    try:
        box = page.locator(input_sel).first
        if await box.is_visible(timeout=1000):
            await box.fill(ans)
            await click_submit_if_present(page)
            await box.press("Enter")
            print(f"[Text] {bot['name']} (C{bot['tier']}) -> '{ans}' ({time.monotonic() - reveal_ts:.2f}s)")
    except Exception as e:
        print(f"[Text Err] {bot['name']}: {e}")


async def submit_slider_bot(bot: dict, slider_pack: dict, reveal_ts: float):
    await wait_window(reveal_ts)
    val = slider_pack["best"] if (bot["tier"] == 1 or random.random() < 0.5) else random.choice(slider_pack["wrong"])
    val_str = str(int(val)) if float(val).is_integer() else str(val)
    page: Page = bot["page"]
    try:
        slider = page.locator('input[type="range"], input[data-functional-selector*="slider"]').first
        if await slider.is_visible(timeout=1000):
            await slider.evaluate(
                """(el, v) => {
                    const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
                    setter.call(el, v);
                    el.dispatchEvent(new Event('input', { bubbles: true }));
                    el.dispatchEvent(new Event('change', { bubbles: true }));
                }""",
                val_str,
            )
            await click_submit_if_present(page)
            print(f"[Slider] {bot['name']} (C{bot['tier']}) -> {val_str} ({time.monotonic() - reveal_ts:.2f}s)")
    except Exception as e:
        print(f"[Slider Err] {bot['name']}: {e}")


async def submit_pin_bot(bot: dict, best_xy: tuple[float, float], reveal_ts: float):
    await wait_window(reveal_ts)
    if bot["tier"] == 1 or random.random() < 0.5:
        x_pct, y_pct = best_xy
    else:
        wrong_pts = [(0.25, 0.25), (0.75, 0.25), (0.5, 0.75)]
        x_pct, y_pct = random.choice(wrong_pts)

    page: Page = bot["page"]
    canvas_sel = '[data-functional-selector*="pin-image"], [data-functional-selector*="drop-pin"], canvas, img'
    try:
        target = page.locator(canvas_sel).first
        box = await target.bounding_box()
        if box:
            await page.mouse.click(box["x"] + box["width"] * x_pct, box["y"] + box["height"] * y_pct)
            await asyncio.sleep(0.1)
            await click_submit_if_present(page)
            print(f"[Pin] {bot['name']} (C{bot['tier']}) -> ({x_pct:.2f}, {y_pct:.2f}) ({time.monotonic() - reveal_ts:.2f}s)")
    except Exception as e:
        print(f"[Pin Err] {bot['name']}: {e}")


# ---------------------------------------------------------------------------
# Join & Main Loop
# ---------------------------------------------------------------------------

async def join_kahoot(context, pin: str, bot_info: dict) -> Page | None:
    page = await context.new_page()
    name, tier = bot_info["name"], bot_info["tier"]
    try:
        await page.goto(f"https://kahoot.it/?pin={pin}", wait_until="domcontentloaded")
        pin_input = page.locator('input[name="gameId"], input[data-functional-selector="game-pin-input"]')
        if await pin_input.is_visible(timeout=5000):
            await pin_input.fill(pin)
            await page.locator('button[type="submit"]').click()

        nick_input = page.locator('input[name="nickname"], input[data-functional-selector="username-input"]')
        await nick_input.wait_for(state="visible", timeout=15000)
        await nick_input.fill(name)
        await page.locator('button[type="submit"], button[data-functional-selector="join-button-username"]').click()
        print(f"[Joined] {name} (Class {tier})")
        return page
    except Exception as e:
        print(f"[Join Failed] {name} (Class {tier}): {e}")
        await page.close()
        return None


async def extract_question_text(lead_page: Page) -> str:
    q_loc = lead_page.locator('[data-functional-selector="block-title"], [data-functional-selector="question-block-title"]')
    if await q_loc.count() > 0:
        return (await q_loc.first.inner_text()).strip()
    return "Select the correct answer:"


async def main():
    roster = build_bot_roster()
    print(f"Starting session for Game PIN: {KAHOOT_PIN} | Total bots: {len(roster)}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        active_bots = []

        for bot_info in roster:
            if bot_info["tier"] == 2:
                await asyncio.sleep(random.uniform(0.5, 1.0))
            context = await browser.new_context()
            page = await join_kahoot(context, KAHOOT_PIN, bot_info)
            if page:
                active_bots.append({**bot_info, "page": page, "context": context})

        if not active_bots:
            print("No bots joined. Exiting.")
            await browser.close()
            return

        lead = active_bots[0]["page"]
        print("All bots ready! Polling for questions...")
        last_sig = None
        idle_ticks = 0

        while idle_ticks < 3600:
            await asyncio.sleep(0.3)

            if "ranking" in lead.url or "podium" in lead.url:
                print("Game finished! Closing bots.")
                break

            ans0 = lead.locator('[data-functional-selector="answer-0"]')
            puzzle_cards = lead.locator('[data-functional-selector*="jumble-card"], [data-functional-selector*="puzzle-card"]')
            text_input = lead.locator(
                'input[data-functional-selector*="text-answer"], textarea[data-functional-selector*="open-ended"]'
            ).first
            slider_input = lead.locator('input[type="range"], input[data-functional-selector*="slider"]').first
            pin_canvas = lead.locator('[data-functional-selector*="pin-image"], [data-functional-selector*="drop-pin"]').first

            # 1. Quiz / True-False / Poll / Multi-Select
            if await ans0.is_visible():
                reveal_ts = time.monotonic()
                q_text = await extract_question_text(lead)
                ms_btn = lead.locator('button[data-functional-selector="multi-select-submit-button"]')
                is_multi = (await ms_btn.count()) > 0

                options = []
                for idx in range(4):
                    loc = lead.locator(f'[data-functional-selector="answer-{idx}"]')
                    if await loc.count() > 0 and await loc.is_visible():
                        options.append((await loc.inner_text()).strip() or f"Option {idx}")

                sig = f"{'MULTI' if is_multi else 'MCQ'}|{q_text}|{'|'.join(options)}"
                if sig != last_sig and options:
                    last_sig = sig
                    idle_ticks = 0
                    if is_multi:
                        best_indices = await solve_multiselect(q_text, options)
                        await asyncio.gather(*(submit_multi_bot(b, best_indices, len(options), reveal_ts) for b in active_bots))
                    else:
                        best_idx = await solve_mcq(q_text, options)
                        await asyncio.gather(*(submit_mcq_bot(b, best_idx, len(options), reveal_ts) for b in active_bots))
                    try:
                        await ans0.wait_for(state="hidden", timeout=60000)
                    except Exception:
                        pass

            # 2. Puzzle / Jumble
            elif await puzzle_cards.count() >= 2 and await puzzle_cards.first.is_visible():
                reveal_ts = time.monotonic()
                q_text = await extract_question_text(lead)
                items = [(await puzzle_cards.nth(i).inner_text()).strip() for i in range(await puzzle_cards.count())]
                sig = f"PUZZLE|{q_text}|{'|'.join(items)}"
                if sig != last_sig and items:
                    last_sig = sig
                    idle_ticks = 0
                    best_order = await solve_puzzle(q_text, items)
                    await asyncio.gather(*(submit_puzzle_bot(b, items, best_order, reveal_ts) for b in active_bots))
                    try:
                        await puzzle_cards.first.wait_for(state="hidden", timeout=60000)
                    except Exception:
                        pass

            # 3. Type Answer / Word Cloud / Open-Ended
            elif await text_input.is_visible():
                reveal_ts = time.monotonic()
                q_text = await extract_question_text(lead)
                sig = f"TEXT|{q_text}"
                if sig != last_sig:
                    last_sig = sig
                    idle_ticks = 0
                    text_pack = await solve_text(q_text)
                    await asyncio.gather(*(submit_text_bot(b, text_pack, reveal_ts) for b in active_bots))
                    try:
                        await text_input.wait_for(state="hidden", timeout=60000)
                    except Exception:
                        pass

            # 4. Slider
            elif await slider_input.is_visible():
                reveal_ts = time.monotonic()
                q_text = await extract_question_text(lead)
                min_v = float((await slider_input.get_attribute("min")) or 0)
                max_v = float((await slider_input.get_attribute("max")) or 100)
                step_attr = await slider_input.get_attribute("step")
                step_v = float(step_attr) if step_attr not in (None, "any", "") else 1.0

                sig = f"SLIDER|{q_text}|{min_v}|{max_v}"
                if sig != last_sig:
                    last_sig = sig
                    idle_ticks = 0
                    slider_pack = await solve_slider(q_text, min_v, max_v, step_v)
                    await asyncio.gather(*(submit_slider_bot(b, slider_pack, reveal_ts) for b in active_bots))
                    try:
                        await slider_input.wait_for(state="hidden", timeout=60000)
                    except Exception:
                        pass

            # 5. Pin on Image / Drop Pin
            elif await pin_canvas.is_visible():
                reveal_ts = time.monotonic()
                q_text = await extract_question_text(lead)
                sig = f"PIN|{q_text}"
                if sig != last_sig:
                    last_sig = sig
                    idle_ticks = 0
                    best_xy = await solve_pin(q_text)
                    await asyncio.gather(*(submit_pin_bot(b, best_xy, reveal_ts) for b in active_bots))
                    try:
                        await pin_canvas.wait_for(state="hidden", timeout=60000)
                    except Exception:
                        pass

            else:
                idle_ticks += 1

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
