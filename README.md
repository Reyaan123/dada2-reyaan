# DADA2 — Agentic RAG over SEC filings

**Fill in both sections at the bottom before you submit.** Keep the deployed
app line exactly in this form; it is how your app gets found and graded.

## Getting started

There is no separate spec. The instructions are in the notebooks, and
`DADA2_00_Start_Here.ipynb` is the one to open first: it explains the
assignment, the setup, the grading, and what each of the four notebooks after
it asks for. Work through them in order.

Get the files the same way as every class day: pull, then work in a copy:

```bash
cd lecture-files
git pull
cd ..
cp -r lecture-files/DADA2 DADA2     # work in the copy, not in lecture-files/
cd DADA2
cp example.env .env                  # then fill in your keys
```

Then open `DADA2_00_Start_Here.ipynb` from your `DADA2` copy and run its
setup check.

Deploying is `DEPLOY.md`.

---

## Submission

```
Deployed app: https://sec-filings-agent-reyaan-lakhani.onrender.com/
```

ADD YOUR NAME TO YOUR RENDER PROJECT SO THE LINK CAN BE PERSONALIZED.

PLEASE ENSURE YOUR URL WORKS BY TESTING IT IN INCOGNITO MODE.

### WHAT TO SUBMIT
Upload the entire project folder (DADA2) via GitHub integration on Canvas

### Write-up

Paste the write-up from notebook 04 (TODO 4.3) here, under the same five
headings.

**Conditions.**
The corpus is 27 section files (Business, Risk Factors and MD&A) from nine 10-K filings, three years each for NVIDIA, AMD and Micron, chunked at 1,200 characters with 150 characters of overlap into 2,124 chunks tagged with source, ticker, fiscal year and section. The chunks are embedded with qwen3-embedding-0.6b on Cloudflare Workers AI (1,024 dimensions) and stored in the Pinecone index dada2-1024. The chat model is gemini-3.5-flash-lite, the decision model is clef-flash on Cloudflare and the reranker is Cohere rerank-v3.5; each search returns k = 5 passages, and the reranking strategy reorders a pool of 10, with an RRF constant of 60. The routers were tested on all 24 labeled questions (14 specific, 10 comparative) and the agents on 14 (8 specific, 6 comparative), with the chat model judging each answer correct or not.

**The routers.**
The decision model was more accurate, routing 20 of 24 questions correctly (83%) against Gemini’s 18 (75%), while Gemini was faster by about a tenth of a second per call (0.88 s against 0.99 s), which is negligible. That two-question gap may not hold, though: the notebook’s reference run had Gemini slightly ahead, and with 24 questions and a non-deterministic chat model the order can flip between runs. What the decision model adds is a probability for each option, which lets route fall back to DEFAULT below SURE = 0.6, plus calls that don’t count against Gemini’s 500-a-day limit. Micron’s R&D question shows what the probabilities add: it came back 0.63 for figure and 0.36 for specific, just over the threshold, so it still routed to figure, but it was a close call that a single label from Gemini wouldn’t show.

**The agents.**
Both agents scored 0.86, but the diagnosis agent made 2.9 chat calls per question (plus 2.4 decision-model calls, which don’t count against the Gemini limit) to the ladder’s 3.9 and took 11.7 s per question to its 12.6 s: the same accuracy for about a quarter fewer Gemini calls, so this is a cost improvement, not an accuracy one. The question where they differed most came from a second run on the six comparative questions: “Which of these companies treat dependence on a small number of customers as a risk?” took the ladder 9 chat calls and diagnosis 2. Because the question names no company, 9 calls is consistent with the ladder’s yes/no grader judging each rung insufficient, so it kept climbing, and each climb cost a grader call plus the search’s own calls. Diagnosis routed it as comparative to hybrid+rrf in one call, the decision model judged those passages sufficient without a Gemini call, and it answered in one more. Across the six comparative questions the totals were 30 chat calls to 16, close to half, so the savings come from comparative questions; specific ones are cheap for both agents.

**What did not help.**
The metadata filter did not help on comparative questions: on a 5-question comparative sample it cut recall@10 from 0.83 without the filter to 0.63 with it, even though the same filter had raised recall@5 on the 6-question specific sample from 0.67 to 1.00. The mechanism is in where_of, which restricts the search to the one company or year the extractor picks and so throws away the other companies’ filings that a comparative question needs. The design routes around it: the router sends comparative questions to hybrid+rrf, and the planner splits them into one filtered search per company.

**What I would ship.**
I would ship the diagnosis agent, because it matched the ladder’s accuracy with fewer Gemini calls. Before shipping I’d want to know the mix of questions users ask: if most are comparative the savings are large, and if most are specific the two agents cost about the same. I’d also want to know how often they ask number questions or about things the filings don’t cover, since the second is where the ladder can waste the most climbs while diagnosis stops after one search (the dress-code question).