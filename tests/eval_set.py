EVAL_SET = [
    # ---------- PTO policy (HR-201) ----------
    {
        "question": "How many sick days do full-time employees get each year?",
        "expected_source": "pto_policy",
        "expected_section": "5. Sick Leave",
        "expected_facts": [["10", "ten"]],
    },
    {
        "question": "How much vacation PTO does an employee with 3 years of service earn per year?",
        "expected_source": "pto_policy",
        "expected_section": "2. Vacation PTO Accrual",
        "expected_facts": [["20", "twenty"]],
    },
    {
        "question": "How many unused vacation days can I carry over into the next year?",
        "expected_source": "pto_policy",
        "expected_section": "3. Carryover and Maximum Balance",
        "expected_facts": [["5", "five"]],
    },
    {
        "question": "How far in advance do I need to request a week of PTO?",
        "expected_source": "pto_policy",
        "expected_section": "4. Requesting PTO",
        "expected_facts": [["two weeks", "2 weeks"]],
    },
    {
        "question": "How long is paid parental leave?",
        "expected_source": "pto_policy",
        "expected_section": "8. Other Leave",
        "expected_facts": [["16"]],
    },
    # ---------- Expense policy (FIN-310) ----------
    {
        "question": "Who needs to approve a $3,000 expense?",
        "expected_source": "workplace_expense_policy",
        "expected_section": "4. Approval Thresholds",
        "expected_facts": [["vice president"]],
    },
    {
        "question": "What is the daily meal limit for international business travel?",
        "expected_source": "workplace_expense_policy",
        "expected_section": "5. Meals",
        "expected_facts": [["100"]],
    },
    {
        "question": "How much will Arasaka reimburse for conferences and courses each year?",
        "expected_source": "workplace_expense_policy",
        "expected_section": "8. Professional Development",
        "expected_facts": [["2,000", "2000"]],
    },
    {
        "question": "How long do I have to submit an expense report?",
        "expected_source": "workplace_expense_policy",
        "expected_section": "10. Submitting Expenses",
        "expected_facts": [["30"]],
    },
    {
        "question": "What is policy FIN-310?",
        "expected_source": "workplace_expense_policy",
        "expected_section": "",
        "expected_facts": [["expense"]],
        "note": "Exact policy code: needs keyword search (Part 3).",
    },
    # ---------- Remote work policy (HR-104), current version is v3 ----------
    {
        "question": "How many sick-day remote work days can I use per calendar year?",
        "expected_source": "remote_work_policy_v3",
        "expected_section": "3. Sick-Day Remote Work Exception",
        "expected_facts": [["10", "ten"]],
    },
    {
        "question": "How many days a week do I need to work from the office?",
        "expected_source": "remote_work_policy_v3",
        "expected_section": "2. Return to Office",
        "expected_facts": [["5", "five"]],
        "note": "Planted data-quality issue (Part 6): outdated v2 says 2 office days.",
    },
    {
        "question": "Can I still get the home office stipend?",
        "expected_source": "remote_work_policy_v3",
        "expected_section": "5. Transition Provisions",
        "expected_facts": [["discontinued", "no longer"]],
        "note": "Planted data-quality issue (Part 6): outdated v2 offers a $500 stipend.",
    },
]
