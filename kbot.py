import asyncio
import json
import os
import random
import re
import time
from typing import Any
import pydantic
from google import genai
from google.genai import types
from playwright.async_api import async_playwright, Page

# 1. Load environment variables first
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
raw_pin_input = os.environ.get("KAHOOT_PIN", "").strip()

pin_match = re.search(r"pin=(\d+)", raw_pin_input)
if pin_match:
    KAHOOT_PIN = pin_match.group(1)
else:
    KAHOOT_PIN = re.sub(r"\D", "", raw_pin_input)

FIRST_CLASS_COUNT = min(max(int(os.environ.get("FIRST_CLASS_COUNT", "1")), 1), 2)
FIRST_CLASS_NAME_1 = os.environ.get("FIRST_CLASS_NAME_1", "AlphaBot").strip() or "AlphaBot"
RAW_FIRST_CLASS_NAME_2 = os.environ.get("FIRST_CLASS_NAME_2", "").strip()

SECOND_CLASS_COUNT = max(int(os.environ.get("SECOND_CLASS_COUNT", "0")), 0)
SECOND_CLASS_BASE_NAME = os.environ.get("SECOND_CLASS_NAME", "RefBot").strip() or "RefBot"

MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.5-pro").strip() or "gemini-2.5-pro"
JOIN_DELAY = float(os.environ.get("JOIN_DELAY", "1.2"))

if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY environment variable. Set it in GitHub Secrets.")

ai_client = genai.Client(api_key=GEMINI_API_KEY)


def build_bot_roster():
    roster = [{"name": FIRST_CLASS_NAME_1, "tier": 1}]
    if FIRST_CLASS_COUNT == 2:
        name_2 = RAW_FIRST_CLASS_NAME_2 if RAW_FIRST_CLASS_NAME_2 else "[default]"
        roster.append({"name": name_2, "tier": 1})
    for i in range(1, SECOND_CLASS_COUNT + 1):
        bot_name = f"{SECOND_CLASS_BASE_NAME}{i}" if SECOND_CLASS_COUNT > 1 else SECOND_CLASS_BASE_NAME
        roster.append({"name": bot_name, "tier": 2})
    return roster


# ---------------------------------------------------------------------------
# Gemini Structured JSON Solvers (Pydantic Schema Enforced)
# ---------------------------------------------------------------------------

async def call_gemini_json(prompt: str, response_schema: type, image_bytes: bytes = None, system_instruction: str = "") -> Any:
    def _generate():
        config_kwargs = {
            "temperature": 0.0,
            "response_mime_type": "application/json",
            "response_schema": response_schema,
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
            
        contents = [prompt]
        if image_bytes:
            contents.append(types.Part.from_bytes(data=image_bytes, mime_type="image/png"))
            
        response = ai_client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs)
        )
        return json.loads(response.text or "{}")

    return await asyncio.to_thread(_generate)


class MCQResponse(pydantic.BaseModel):
    correct_index: int


async def solve_mcq(question: str, options: list[str], image_bytes: bytes = None) -> int:
    opts_str = "\n".join([f"Index [{i}]: {opt}" for i, opt in enumerate(options)])
    prompt = (
        f"Analyze this Kahoot question carefully.\n"
        f"Question: {question}\n\n"
        f"Available Options:\n{opts_str}\n\n"
        f"Select the correct option index (0 to {len(options)-1})."
    )
    try:
        result = await call_gemini_json(
            prompt, 
            response_schema=MCQResponse, 
            image_bytes=image_bytes, 
            system_instruction="You are an expert academic trivia solver. Accurately identify the correct option index."
        )
        idx = int(result.get("correct_index", 0))
        if 0 <= idx < len(options):
            print(f"[AlphaBot Match] Chose Index {idx} -> {options[idx]}")
            return idx
    except Exception as e:
        print(f"[Gemini Structured MCQ Error] {e}")
    return 0


class MultiResponse(pydantic.BaseModel):
    indices: list[int]


async def solve_multiselect(question: str, options: list[str], image_bytes: bytes = None) -> list[int]:
    opts = "\n".join(f"Index [{i}]: {o}" for i, o in enumerate(options))
    prompt = f"Q: {question}\n{opts}\nSelect all correct option indices."
    try:
        result = await call_gemini_json(prompt, response_schema=MultiResponse, image_bytes=image_bytes, system_instruction="Output valid indices array.")
        valid = sorted({int(i) for i in result.get("indices", [0]) if 0 <= int(i) < len(options)})
        return valid if valid else [0]
    except Exception as e:
        print(f"[Gemini Multi Error] {e}")
    return [0]


class PuzzleResponse(pydantic.BaseModel):
    order: list[int]


async def solve_puzzle(question: str, items: list[str], image_bytes: bytes = None) -> list[int]:
    opts = "\n".join(f"Index [{i}]: {o}" for i, o in enumerate(items))
    n = len(items)
    prompt = f"Q: {question}\n{opts}\nOrder items correctly by index."
    try:
        result = await call_gemini_json(prompt, response_schema=PuzzleResponse, image_bytes=image_bytes, system_instruction="Output correct sequence order.")
        order = [int(i) for i in result.get("order", []) if 0 <= int(i) < n]
        if len(order) == n and len(set(order)) == n:
            return order
    except Exception as e:
        print(f"[Gemini Puzzle Error] {e}")
    return list(range(n))


class TextResponse(pydantic.BaseModel):
    best_answer: str
    wrong_alternatives: list[str]


async def solve_text(question: str, image_bytes: bytes = None) -> dict:
    prompt = f"Q: {question}\nProvide the best short answer (under 20 chars) and 3 plausible wrong answers."
    try:
        result = await call_gemini_json(prompt, response_schema=TextResponse, image_bytes=image_bytes, system_instruction="Provide concise text answers.")
        best = str(result.get("best_answer", "Yes"))[:20].strip()
        wrong = [str(w)[:20].strip() for w in result.get("wrong_alternatives", ["No", "Maybe", "None"])[:3]]
        while len(wrong) < 3:
            wrong.append("Unknown")
        return {"best": best, "wrong": wrong}
    except Exception as e:
        print(f"[Gemini Text Error] {e}")
    return {"best": "Yes", "wrong": ["No", "Maybe", "Unknown"]}


class SliderResponse(pydantic.BaseModel):
    value: float


async def solve_slider(question: str, min_v: float, max_v: float, step: float, image_bytes: bytes = None) -> dict:
    def snap(val: float) -> float:
        clamped = max(min_v, min(max_v, float(val)))
        if step > 0:
            clamped = min_v + round((clamped - min_v) / step) * step
        return round(clamped, 4)

    best = snap((min_v + max_v) / 2)
    prompt = f"Q: {question}\nRange:[{min_v},{max_v}]\nEstimate target numeric value."
    try:
        result = await call_gemini_json(prompt, response_schema=SliderResponse, image_bytes=image_bytes, system_instruction="Provide numeric slider value.")
        best = snap(result.get("value", best))
    except Exception as e:
        print(f"[Gemini Slider Error] {e}")

    span = max_v - min_v
    wrong = [snap(best + off) for off in [-0.25 * span, 0.2 * span, 0.4 * span]]
    return {"best": best, "wrong": wrong}


class PinResponse(pydantic.BaseModel):
    x: float
    y: float


async def solve_pin(question: str, image_bytes: bytes = None) -> tuple[float, float]:
    prompt = f"Q: {question}\nEstimate target image coordinates (0.1 to 0.9)."
    try:
        result = await call_gemini_json(prompt, response_schema=PinResponse, image_bytes=image_bytes, system_instruction="Provide coordinate map x and y.")
        x = max(0.1, min(0.9, float(result.get("x", 0.5))))
        y = max(0.1, min(0.9, float(result.get("y", 0.5))))
        return (x, y)
    except Exception:
        return (0.5, 0.5)


# ---------------------------------------------------------------------------
# Bot Action Dispatchers (Tier 1 = Instant Speed | Tier 2 = Slow Reference)
# ---------------------------------------------------------------------------

async def wait_window(tier: int, reveal_timestamp: float):
    if tier == 1:
        target_delay = random.uniform(0.1, 0.3)
    else:
        target_delay = random.uniform(1.5, 3.0)
    await asyncio.sleep(max(0.0, target_delay - (time.monotonic() - reveal_timestamp)))


async def click_submit_if_present(page: Page):
    submit_sel = 'button[data-functional-selector*="submit"], button[type="submit"]'
    btn = page.locator(submit_sel).first
    if await btn.count() > 0 and await btn.is_visible(timeout=800):
        await btn.click()


async def submit_mcq_bot(bot: dict, best_idx: int, n: int, reveal_ts: float):
    await wait_window(bot["tier"], reveal_ts)
    idx = best_idx if bot["tier"] == 1 or random.random() < 0.5 else random.choice([i for i in range(n) if i != best_idx])
    try:
        btn = bot["page"].locator(f'[data-functional-selector="answer-{idx}"]')
        if await btn.is_visible(timeout=1000):
            await btn.click()
    except Exception:
        pass


async def submit_multi_bot(bot: dict, best_indices: list[int], n: int, reveal_ts: float):
    await wait_window(bot["tier"], reveal_ts)
    indices = best_indices if bot["tier"] == 1 or random.random() < 0.5 else [0]
    try:
        for idx in indices:
            btn = bot["page"].locator(f'[data-functional-selector="answer-{idx}"]')
            if await btn.is_visible(timeout=800):
                await btn.click()
                await asyncio.sleep(0.04)
        await click_submit_if_present(bot["page"])
    except Exception:
        pass


async def submit_puzzle_bot(bot: dict, items: list[str], best_order: list[int], reveal_ts: float):
    await wait_window(bot["tier"], reveal_ts)
    target_order = best_order if bot["tier"] == 1 or random.random() < 0.5 else best_order[::-1]
    page: Page = bot["page"]
    card_sel = '[data-functional-selector*="jumble-card"], [data-functional-selector*="puzzle-card"], [draggable="true"]'
    try:
        for target_pos, orig_idx in enumerate(target_order):
            target_text = items[orig_idx]
            cards = page.locator(card_sel)
            count = await cards.count()
            for cur_pos in range(count):
                txt = (await cards.nth(cur_pos).inner_text()).strip()
                if txt == target_text and cur_pos != target_pos:
                    src_box = await cards.nth(cur_pos).bounding_box()
                    dst_box = await cards.nth(target_pos).bounding_box()
                    if src_box and dst_box:
                        await page.mouse.move(src_box["x"] + src_box["width"] / 2, src_box["y"] + src_box["height"] / 2)
                        await page.mouse.down()
                        await page.mouse.move(dst_box["x"] + dst_box["width"] / 2, dst_box["y"] + dst_box["height"] / 2, steps=4)
                        await page.mouse.up()
                    break
        await click_submit_if_present(page)
    except Exception:
        pass


async def submit_text_bot(bot: dict, text_pack: dict, reveal_ts: float):
    await wait_window(bot["tier"], reveal_ts)
    ans = text_pack["best"] if (bot["tier"] == 1 or random.random() < 0.5) else random.choice(text_pack["wrong"])
    page: Page = bot["page"]
    input_sel = 'input[data-functional-selector*="text-answer"], textarea[data-functional-selector*="open-ended"], input[placeholder*="answer" i], textarea'
    try:
        box = page.locator(input_sel).first
        if await box.is_visible(timeout=1000):
            await box.fill(ans)
            await click_submit_if_present(page)
            await box.press("Enter")
    except Exception:
        pass


async def submit_slider_bot(bot: dict, slider_pack: dict, reveal_ts: float):
    await wait_window(bot["tier"], reveal_ts)
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
    except Exception:
        pass


async def submit_pin_bot(bot: dict, best_xy: tuple[float, float], reveal_ts: float):
    await wait_window(bot["tier"], reveal_ts)
    x_pct, y_pct = best_xy if bot["tier"] == 1 or random.random() < 0.5 else (0.25, 0.25)
    page: Page = bot["page"]
    canvas_sel = '[data-functional-selector*="pin-image"], [data-functional-selector*="drop-pin"], canvas, img'
    try:
        target = page.locator(canvas_sel).first
        box = await target.bounding_box()
        if box:
            await page.mouse.click(box["x"] + box["width"] * x_pct, box["y"] + box["height"] * y_pct)
            await asyncio.sleep(0.05)
            await click_submit_if_present(page)
    except Exception:
        pass


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
    try:
        q_loc = lead_page.locator('[data-functional-selector="block-title"], [data-functional-selector="question-block-title"], h2')
        if await q_loc.count() > 0:
            text = (await q_loc.first.inner_text()).strip()
            if text:
                return text
    except Exception:
        pass
    return "Select the correct answer:"


async def has_question_image(lead_page: Page) -> bool:
    """Detects if an image, graph, or media element is present in the question layout."""
    try:
        media_loc = lead_page.locator('[data-functional-selector*="question-media"], img[data-functional-selector*="image"], .question-media img, video')
        return (await media_loc.count() > 0) and (await media_loc.first.is_visible())
    except Exception:
        return False


async def main():
    roster = build_bot_roster()
    print(f"Starting session for Game PIN: {KAHOOT_PIN} | Total bots: {len(roster)}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        active_bots = []

        # Safe stagger interval (1.2s) with auto-retry to ensure 100% join rate
        for idx, bot_info in enumerate(roster):
            if idx > 0:
                await asyncio.sleep(JOIN_DELAY)

            context = await browser.new_context()
            page = None
            for attempt in range(2):
                page = await join_kahoot(context, KAHOOT_PIN, bot_info)
                if page:
                    break
                print(f"[Retry] Retrying join for {bot_info['name']}...")
                await asyncio.sleep(1.5)

            if page:
                active_bots.append({**bot_info, "page": page, "context": context})

        if not active_bots:
            print("No bots joined. Exiting.")
            await browser.close()
            return

        print(f"Successfully connected {len(active_bots)}/{len(roster)} bots! Polling for questions...")
        lead = active_bots[0]["page"]
        last_sig = None
        idle_ticks = 0

        while idle_ticks < 3600:
            await asyncio.sleep(0.2)

            if "ranking" in lead.url or "podium" in lead.url:
                print("Game finished! Closing bots.")
                break

            ans0 = lead.locator('[data-functional-selector="answer-0"]')
            puzzle_cards = lead.locator('[data-functional-selector*="jumble-card"], [data-functional-selector*="puzzle-card"]')
            text_input = lead.locator('input[data-functional-selector*="text-answer"], textarea[data-functional-selector*="open-ended"]').first
            slider_input = lead.locator('input[type="range"], input[data-functional-selector*="slider"]').first
            pin_canvas = lead.locator('[data-functional-selector*="pin-image"], [data-functional-selector*="drop-pin"]').first

            if await ans0.is_visible():
                reveal_ts = time.monotonic()
                q_text = await extract_question_text(lead)
                
                # Conditional Vision: Only capture screenshot if an image/graph is actually present!
                question_screenshot = await lead.screenshot() if await has_question_image(lead) else None

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
                        best_indices = await solve_multiselect(q_text, options, image_bytes=question_screenshot)
                        await asyncio.gather(*(submit_multi_bot(b, best_indices, len(options), reveal_ts) for b in active_bots))
                    else:
                        best_idx = await solve_mcq(q_text, options, image_bytes=question_screenshot)
                        await asyncio.gather(*(submit_mcq_bot(b, best_idx, len(options), reveal_ts) for b in active_bots))
                    
                    # Safety Cooldown Pause to prevent rate-limiting & 429 errors
                    await asyncio.sleep(1.0)
                    
                    try:
                        await ans0.wait_for(state="hidden", timeout=60000)
                    except Exception:
                        pass

            elif await puzzle_cards.count() >= 2 and await puzzle_cards.first.is_visible():
                reveal_ts = time.monotonic()
                q_text = await extract_question_text(lead)
                question_screenshot = await lead.screenshot() if await has_question_image(lead) else None
                items = [(await puzzle_cards.nth(i).inner_text()).strip() for i in range(await puzzle_cards.count())]
                sig = f"PUZZLE|{q_text}|{'|'.join(items)}"
                if sig != last_sig and items:
                    last_sig = sig
                    idle_ticks = 0
                    best_order = await solve_puzzle(q_text, items, image_bytes=question_screenshot)
                    await asyncio.gather(*(submit_puzzle_bot(b, items, best_order, reveal_ts) for b in active_bots))
                    await asyncio.sleep(1.0)
                    try:
                        await puzzle_cards.first.wait_for(state="hidden", timeout=60000)
                    except Exception:
                        pass

            elif await text_input.is_visible():
                reveal_ts = time.monotonic()
                q_text = await extract_question_text(lead)
                question_screenshot = await lead.screenshot() if await has_question_image(lead) else None
                sig = f"TEXT|{q_text}"
                if sig != last_sig:
                    last_sig = sig
                    idle_ticks = 0
                    text_pack = await solve_text(q_text, image_bytes=question_screenshot)
                    await asyncio.gather(*(submit_text_bot(b, text_pack, reveal_ts) for b in active_bots))
                    await asyncio.sleep(1.0)
                    try:
                        await text_input.wait_for(state="hidden", timeout=60000)
                    except Exception:
                        pass

            elif await slider_input.is_visible():
                reveal_ts = time.monotonic()
                q_text = await extract_question_text(lead)
                question_screenshot = await lead.screenshot() if await has_question_image(lead) else None
                min_v = float((await slider_input.get_attribute("min")) or 0)
                max_v = float((await slider_input.get_attribute("max")) or 100)
                step_attr = await slider_input.get_attribute("step")
                step_v = float(step_attr) if step_attr not in (None, "any", "") else 1.0

                sig = f"SLIDER|{q_text}|{min_v}|{max_v}"
                if sig != last_sig:
                    last_sig = sig
                    idle_ticks = 0
                    slider_pack = await solve_slider(q_text, min_v, max_v, step_v, image_bytes=question_screenshot)
                    await asyncio.gather(*(submit_slider_bot(b, slider_pack, reveal_ts) for b in active_bots))
                    await asyncio.sleep(1.0)
                    try:
                        await slider_input.wait_for(state="hidden", timeout=60000)
                    except Exception:
                        pass

            elif await pin_canvas.is_visible():
                reveal_ts = time.monotonic()
                q_text = await extract_question_text(lead)
                question_screenshot = await lead.screenshot() if await has_question_image(lead) else None
                sig = f"PIN|{q_text}"
                if sig != last_sig:
                    last_sig = sig
                    idle_ticks = 0
                    best_xy = await solve_pin(q_text, image_bytes=question_screenshot)
                    await asyncio.gather(*(submit_pin_bot(b, best_xy, reveal_ts) for b in active_bots))
                    await asyncio.sleep(1.0)
                    try:
                        await pin_canvas.wait_for(state="hidden", timeout=60000)
                    except Exception:
                        pass

            else:
                idle_ticks += 1

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
