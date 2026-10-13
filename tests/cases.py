"""A few hand-written decisions of every question type and several lengths (parity smoke tests)."""

TEAM = {"type": "choice", "instructions": "Which team should handle this?",
        "criteria": {"billing": "Charges, refunds, invoices", "technical": "App or site faults",
                     "fraud": "Suspected unauthorised use"}}
REFUND = {"type": "noul", "instructions": "Is the customer asking for a refund?"}
URGENCY = {"type": "score", "instructions": "How urgent is this?",
           "criteria": ["Can wait", "Today", "Blocking the customer now"]}
LONG = ("Hi team, following up on ticket 48213. Since the March update the mobile app signs me out every time I "
        "switch networks, and after signing back in the invoices tab shows last year's totals. I also noticed two "
        "charges of 49.99 on the 3rd, one of which I never authorised; my bank says it came from a device in another "
        "country. I need the duplicate reversed, the account locked until I can change my password, and somebody to "
        "tell me whether my saved cards were exposed. I have a client demo on Thursday and cannot use the product "
        "until this is sorted. ") * 3
TOOLS = {"type": "choice", "instructions": "Which tool should the agent call next?",
         "criteria": {f"tool_{i}": d for i, d in enumerate([
             "Search the knowledge base", "Open a refund", "Lock the account", "Escalate to a human",
             "Send a password reset", "Look up an order", "Schedule a call back", "Close the ticket",
             "Request a screenshot", "Check service status", "Update billing details", "Answer directly"])}}

CASES = [
    ("I was charged twice this month, please refund one of them.", REFUND),
    ("I was charged twice this month, please refund one of them.", TEAM),
    ("I was charged twice this month, please refund one of them.", URGENCY),
    ("The app crashes when I open settings.", TEAM),
    ("Where is my parcel? It was due Monday.", REFUND),
    ({"user": "amelie", "plan": "pro", "message": "Someone logged in from Lagos and bought gift cards."}, TEAM),
    (LONG, TEAM),
    (LONG, URGENCY),
    (LONG, TOOLS),
    ("Merci de me rembourser le deuxième prélèvement.", REFUND),
]
