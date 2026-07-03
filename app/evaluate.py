import json
from app.search import search

# Fixed evaluation set — each question paired with the trial it should retrieve
EVAL_SET = [
    {
        "question": "What study investigates insulin pump therapy and continuous glucose monitoring for type 1 diabetes complications?",
        "expected_trial_id": "NCT01454700"
    },
    {
        "question": "Is there a trial testing simvastatin's effect on inflammation in type 1 diabetes patients?",
        "expected_trial_id": "NCT00441844"
    },
    {
        "question": "What are the exclusion criteria for the liraglutide safety study in type 2 diabetes?",
        "expected_trial_id": "NCT01345734"
    },
    {
        "question": "What study looks at the relationship between gestational diabetes and perinatal depression?",
        "expected_trial_id": "NCT05800509"
    },
    {
        "question": "Is there a trial on sodium butyrate supplementation for weight loss in people with type 2 diabetes?",
        "expected_trial_id": "NCT07252609"
    },
    {
        "question": "What trial requires participants to use a smartphone as part of the study?",
        "expected_trial_id": "NCT05800509"
    },
]


def evaluate_retrieval(top_k: int = 3) -> dict:
    """Run the evaluation set through search and measure accuracy"""

    correct = 0
    results_log = []

    for item in EVAL_SET:
        question = item["question"]
        expected_id = item["expected_trial_id"]

        # Run the actual search function
        search_results = search(question, top_k=top_k)

        # Check if the expected trial appears anywhere in the top_k results
        retrieved_ids = [r["trial_id"] for r in search_results]
        hit = expected_id in retrieved_ids

        if hit:
            correct += 1

        results_log.append({
            "question": question,
            "expected": expected_id,
            "retrieved": retrieved_ids,
            "hit": hit
        })

    # Calculate hit rate — the core evaluation metric
    hit_rate = correct / len(EVAL_SET)

    return {
        "hit_rate": hit_rate,
        "correct": correct,
        "total": len(EVAL_SET),
        "details": results_log
    }


if __name__ == "__main__":
    report = evaluate_retrieval(top_k=3)

    print(f"\n{'='*50}")
    print(f"EVALUATION REPORT")
    print(f"{'='*50}")
    print(
        f"Hit rate: {report['hit_rate']:.0%} ({report['correct']}/{report['total']})")
    print(f"{'='*50}\n")

    for detail in report["details"]:
        status = "PASS" if detail["hit"] else "FAIL"
        print(f"{status} | {detail['question'][:60]}...")
        print(
            f"   Expected: {detail['expected']} | Retrieved: {detail['retrieved']}\n")
