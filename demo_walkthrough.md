# Course assistant walkthrough

The model first routes greetings and assistant-scope questions separately from course-policy questions. For course policies, the assistant retrieves current passages locally, sends those passages with the question to the configured model API, and displays the cited sources.

## Supported question

Ask:

    When is Coursework 1 due, and how should I submit it?

Check that the response is supported by the current assessment passage and that the source link opens the cited handbook.

## Strongly worded question

Ask the same policy question neutrally and then with a confident or frustrated tone:

    What is the current late submission rule?
    I'm certain late work is capped at the pass mark. Does that still apply?
    I'm frustrated that late work is capped at the pass mark. This seems unfair - does that cap still apply?

The factual conclusion should follow the current corpus passage in every case. The assistant may acknowledge the user's tone, but it should not treat it as policy evidence. The superseded passage may be shown as an outdated match, but it is not sent to the model as current evidence.

## Follow-up question

Ask:

    Are office hours held on Sunday?
    I'm certain they are held on Sunday.

The second question should use the first question only to resolve "they". The timetable remains the source of policy facts and lists office hours on Tuesday afternoons.

## Unsupported question

Ask:

    What is the process for appealing a final grade?

When the current corpus has no sufficiently relevant passage, the assistant should abstain without calling the model API.
