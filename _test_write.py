import sys
sys.path.insert(0, r"D:\VORTEX_FLAME")
from soul_memory import write

test_id = write(
    soul="einstein",
    category="domain_memory",
    content={
        "topic": "Test: Capybara index",
        "source": "capybara_dataset",
        "question": "What is the speed of light?",
        "answer": "299,792,458 m/s",
        "domain": "physics",
    },
    importance=0.5,
    tags=["test", "capybara", "physics"],
)
print(f"Test write OK: {test_id}")
