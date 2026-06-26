import os
import json
import time
from google import genai
from supabase import create_client, Client

# ─────────────────────────────────────────
# 1. CONNECTIONS
# ─────────────────────────────────────────

client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY"),
)

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ─────────────────────────────────────────
# 2. AGENT IDENTITY
# ─────────────────────────────────────────

SYSTEM_INSTRUCTION = """
You are a data triage agent for Vocabeck, a German language learning platform.

Your job is to classify user feedback reports into exactly one of three categories:
- Translation (problems with the English translation or CEFR level assignment)
- Audio (problems with pronunciation, audio clips, or sound quality)
- Spelling (problems with typos, capitalization, or grammar in the German text)

When given a user comment, respond with ONLY a valid JSON object in this exact format:
{"category": "Translation", "confidence": "high", "reason": "one sentence explanation"}

Do not add any text before or after the JSON. No markdown. No backticks. Just the raw JSON.
"""

# ─────────────────────────────────────────
# 3. TOOL 1 — Fetch new feedback
# ─────────────────────────────────────────

def fetch_new_feedback():
    print("📥 Fetching new feedback from Supabase...")
    response = supabase.table("user_feedback") \
        .select("feedback_id, word_id, issue_type, user_comment") \
        .eq("status", "New") \
        .execute()
    print(f"   Found {len(response.data)} new feedback items.")
    return response.data

# ─────────────────────────────────────────
# 4. TOOL 2 — Classify with Gemini
# ─────────────────────────────────────────

def classify_comment(comment: str) -> dict:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f'user_comment: "{comment}"',
        config={
            "system_instruction": SYSTEM_INSTRUCTION,
            "temperature": 0.1,
            "max_output_tokens": 256,
        }
    )
    raw = response.text.strip()
    return json.loads(raw)

# ─────────────────────────────────────────
# 5. TOOL 3 — Update status
# ─────────────────────────────────────────

def mark_as_under_review(feedback_id: int):
    supabase.table("user_feedback") \
        .update({"status": "Under Review"}) \
        .eq("feedback_id", feedback_id) \
        .execute()

# ─────────────────────────────────────────
# 6. THE AGENT LOOP
# ─────────────────────────────────────────

def run_triage_agent():
    print("\n🤖 Vocabeck Triage Agent starting...\n")

    feedback_items = fetch_new_feedback()

    if not feedback_items:
        print("✅ No new feedback to process. All clear.")
        return

    queues = {
        "Translation": [],
        "Audio": [],
        "Spelling": []
    }

    for item in feedback_items:
        comment = item["user_comment"]
        feedback_id = item["feedback_id"]

        print(f"🔍 Processing feedback_id {feedback_id}: \"{comment[:50]}...\"")

        result = classify_comment(comment)
        category = result["category"]
        reason = result["reason"]

        queues[category].append({
            "feedback_id": feedback_id,
            "word_id": item["word_id"],
            "comment": comment,
            "reason": reason
        })

        mark_as_under_review(feedback_id)
        print(f"   ✓ Routed to {category} queue. Status updated to 'Under Review'.")
        print(f"   ⏳ Waiting 15 seconds for rate limit...")
        time.sleep(15)

    print("\n" + "="*50)
    print("📋 TRIAGE COMPLETE — TEAM QUEUES")
    print("="*50)

    for team, items in queues.items():
        print(f"\n🗂  {team} Team ({len(items)} items):")
        for i in items:
            print(f"   • feedback_id {i['feedback_id']} | word_id {i['word_id']}")
            print(f"     Comment: {i['comment']}")
            print(f"     Reason:  {i['reason']}")

    print("\n✅ Agent run complete.")

# ─────────────────────────────────────────
# 7. RUN
# ─────────────────────────────────────────

if __name__ == "__main__":
    run_triage_agent()