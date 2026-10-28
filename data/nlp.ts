export interface Instructor {
  name: string;
  url?: string;
}

export interface ScheduleRow {
  week: number;
  date: string;
  topic: string;
  materials?: string;
  due?: string;
  /**
   * Hide this row's readings and papers on the public schedule at /nlp.
   * They stay visible on /nlp/planning, which is the full instructor view.
   * Set this while a week's materials are still being finalised.
   */
  materialsHidden?: boolean;
  /**
   * Hide only this row's suggested papers on /nlp, leaving its readings
   * visible. Set from `papersVisibleThrough`, so a week's readings can go out
   * while the paper list is still being settled.
   */
  papersHidden?: boolean;
  /**
   * Lecture slides. When set, the topic becomes a link that opens the PDF in a
   * new tab. Files live in public/slides/.
   */
  slides?: string;
  /**
   * Fuller topic text shown only on /nlp/planning. Use when the public title
   * is deliberately abbreviated ("Post-Training (e.g. RLHF)") but you want the
   * complete list of subtopics in your own view.
   */
  planningTopic?: string;
}

export const courseInfo = {
  title: "Natural Language Processing",
  number: "CSCI 680",
  section: "04",
  institution: "William & Mary",
  semester: "Fall 2026",
  time: "MWF 10:00–10:50 AM",
  location: "Integrated Science Center (ISC) 3280",
  description:
    "This course provides a comprehensive introduction to natural language processing, spanning foundational techniques through large language models. The first half of the course will focus on linguistic and statistical fundamentals before advancing to neural architectures. The second half of the course will focus on the modern LLM pipeline (e.g. pre-training, post-training, and prompting). Students will also engage with advanced topics and emerging areas in NLP (e.g. interpretability, harms and risks of language modeling, data-centric NLP).",
  instructors: [
    { name: "Stephanie Schoch", url: "/" },
  ] as Instructor[],
  officeHours: "M 1:00–2:00 PM",
  syllabus: "/csci_680_nlp_fa26.pdf",
  prerequisites: "Students should be proficient in Python. Experience with packages such as SciPy, Scikit-learn, and PyTorch is helpful. Students should also have experience with Calculus, Linear Algebra, and Probability & Statistics.",
};

export interface DeadlineRow {
  /** Week the item is due. */
  week: number;
  deadline: string;
  /** Release date, for multi-day assignments. Omit for quizzes and in-class items. */
  released?: string;
  /** Due date. */
  date: string;
  time?: string;
}

// Ordered by due date.
export const deadlines: DeadlineRow[] = [
  { week: 4, deadline: "Project Pitch Slide", date: "Mon 09/14", time: "11:59 PM" },
  { week: 5, deadline: "Quiz 1: Statistical Foundations", date: "Mon 09/21", time: "10:00 AM" },
  { week: 5, deadline: "Project Team Formation", date: "Wed 09/23", time: "11:59 PM" },
  { week: 6, deadline: "Homework 1", released: "Mon 09/14", date: "Wed 09/30", time: "11:59 PM" },
  { week: 6, deadline: "Project Proposal", date: "Fri 10/02", time: "11:59 PM" },
  { week: 8, deadline: "Quiz 2: Neural Methods", date: "Wed 10/14", time: "10:00 AM" },
  { week: 9, deadline: "Homework 2", released: "Wed 09/30", date: "Fri 10/23", time: "11:59 PM" },
  { week: 10, deadline: "Quiz 3: LLMs", date: "Fri 10/30", time: "10:00 AM" },
  { week: 11, deadline: "Project Progress Report", date: "Mon 11/02", time: "11:59 PM" },
  { week: 13, deadline: "Homework 3", released: "Mon 11/02", date: "Fri 11/20", time: "11:59 PM" },
  { week: 14, deadline: "Project Final Report", date: "Tue 11/24", time: "11:59 PM" },
];

export interface Resource {
  title: string;
  url: string;
  description?: string;
}

export const resources: Resource[] = [
  {
    title: "ACL Style Files",
    url: "https://github.com/acl-org/acl-style-files",
    description:
      "LaTeX template and style files used for the project proposal, progress report, and final report.",
  },
];

// Jurafsky & Martin, Speech and Language Processing, 3rd ed. draft.
// Chapter numbers are from the Aug 19 2026 release — this is a living draft and
// chapters do move between releases, so re-verify each semester.
const SLP = "https://web.stanford.edu/~jurafsky/slp3";
// Rendered as "J&M: §7", or "J&M: §3, §13" when a day has several. The optional
// `sec` narrows a citation to a section ("7.6"); it still links the whole
// chapter PDF.
const jm = (ch: number, sec?: string) => `[§${sec ?? ch}](${SLP}/${ch}.pdf)`;

// The "J&M:" prefix goes *inside* the first J&M link so the whole reference is
// clickable — otherwise the prefix renders in body colour beside accent-coloured
// links and reads like a dead link. Other refs (papers) are left as they are, so
// a day whose only reading is a paper gets no prefix.
const readings = (...refs: string[]) => {
  const i = refs.findIndex((r) => r.startsWith("[§"));
  return `Readings: ${refs.map((r, j) => (j === i ? r.replace(/^\[/, "[J&M: ") : r)).join(", ")}`;
};

// [short display name, url, full title]. The short name is what appears in the
// schedule; the full title becomes the link's hover tooltip.
type Paper = readonly [short: string, url: string, full: string];
// Exported so ScheduleTable can find — and drop — this line when a row's
// papers are hidden but its readings are not.
export const PAPERS_LABEL = "Suggested Papers:";
const paper = ([s, u, f]: Paper) => `[${s}](${u} "${f}")`;
const papers = (...ps: Paper[]) => `${PAPERS_LABEL} ${ps.map(paper).join(", ")}`;

// Linked notebooks live in public/notebooks/. Rendered as a bare "Notebook"
// link on its own line, under the readings.
const notebook = (file: string) => `[Notebook](${file})`;

// Handout PDFs live in public/handouts/. Rendered as a bare link, named after the
// handout, on its own line under the readings.
const handout = (title: string, file: string) => `[${title}](${file})`;

// Papers students may read but are not required to, on their own "Optional:"
// line under the readings. Unlike suggested papers, these are shown as soon as
// the row's readings are.
const optional = (...ps: Paper[]) => `Optional: ${ps.map(paper).join(", ")}`;

// Each part is its own line in the rendered cell. See toHtml in
// components/ScheduleTable.tsx.
const material = (...parts: string[]) => parts.join("\n");

const P = {
  word2vec: ["word2vec", "https://arxiv.org/abs/1301.3781",
    "Efficient Estimation of Word Representations in Vector Space"],
  glove: ["GloVe", "https://aclanthology.org/D14-1162/",
    "GloVe: Global Vectors for Word Representation"],
  seq2seq: ["Seq2Seq", "https://arxiv.org/abs/1409.3215",
    "Sequence to Sequence Learning with Neural Networks"],
  bahdanau: ["Bahdanau Attention", "https://arxiv.org/abs/1409.0473",
    "Neural Machine Translation by Jointly Learning to Align and Translate"],
  attention: ["Attention Is All You Need", "https://arxiv.org/abs/1706.03762",
    "Attention Is All You Need"],
  bert: ["BERT", "https://aclanthology.org/N19-1423/",
    "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"],
  gpt1: ["GPT", "https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf",
    "Improving Language Understanding by Generative Pre-Training"],
  bpe: ["BPE / Subword Units", "https://aclanthology.org/P16-1162/",
    "Neural Machine Translation of Rare Words with Subword Units"],
  scaling: ["Scaling Laws", "https://arxiv.org/abs/2001.08361",
    "Scaling Laws for Neural Language Models"],
  nucleus: ["Nucleus Sampling", "https://arxiv.org/abs/1904.09751",
    "The Curious Case of Neural Text Degeneration"],
  bleu: ["BLEU", "https://aclanthology.org/P02-1040/",
    "BLEU: a Method for Automatic Evaluation of Machine Translation"],
  mauve: ["MAUVE", "https://arxiv.org/abs/2102.01454",
    "MAUVE: Measuring the Gap Between Neural Text and Human Text"],
  llama3: ["Llama 3", "https://arxiv.org/abs/2407.21783",
    "The Llama 3 Herd of Models"],
  instructgpt: ["InstructGPT", "https://arxiv.org/abs/2203.02155",
    "Training Language Models to Follow Instructions with Human Feedback"],
  dpo: ["DPO", "https://arxiv.org/abs/2305.18290",
    "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"],
  lora: ["LoRA", "https://arxiv.org/abs/2106.09685",
    "LoRA: Low-Rank Adaptation of Large Language Models"],
  adapters: ["Adapters", "https://arxiv.org/abs/1902.00751",
    "Parameter-Efficient Transfer Learning for NLP"],
  gpt3: ["GPT-3", "https://arxiv.org/abs/2005.14165",
    "Language Models are Few-Shot Learners"],
  cot: ["Chain-of-Thought", "https://arxiv.org/abs/2201.11903",
    "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"],
  r1: ["DeepSeek-R1", "https://arxiv.org/abs/2501.12948",
    "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"],
  helm: ["HELM", "https://arxiv.org/abs/2211.09110",
    "Holistic Evaluation of Language Models"],
  mmlu: ["MMLU", "https://arxiv.org/abs/2009.03300",
    "Measuring Massive Multitask Language Understanding"],
  arena: ["Chatbot Arena", "https://arxiv.org/abs/2403.04132",
    "Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference"],
  llmJudge: ["LLM-as-a-Judge", "https://openreview.net/pdf?id=uccHPGDlao",
    "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"],
  livebench: ["LiveBench", "https://openreview.net/forum?id=sKYHBTAxVa",
    "LiveBench: A Challenging, Contamination-Limited LLM Benchmark"],
  rag: ["RAG", "https://arxiv.org/abs/2005.11401",
    "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"],
  react: ["ReAct", "https://arxiv.org/abs/2210.03629",
    "ReAct: Synergizing Reasoning and Acting in Language Models"],
  toolformer: ["Toolformer", "https://arxiv.org/abs/2302.04761",
    "Toolformer: Language Models Can Teach Themselves to Use Tools"],
  // ACM returns 403 to automated requests; the page loads fine in a browser.
  parrots: ["Stochastic Parrots", "https://dl.acm.org/doi/10.1145/3442188.3445922",
    "On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?"],
  risks: ["Social Risks of Harm", "https://arxiv.org/abs/2112.04359",
    "Ethical and Social Risks of Harm from Language Models"],
  attnNotExpl: ["Attention is not Explanation", "https://aclanthology.org/N19-1357/",
    "Attention is not Explanation"],
  attnNotNotExpl: ["Attention is not not Explanation", "https://aclanthology.org/D19-1002/",
    "Attention is not not Explanation"],
  checklist: ["CheckList", "https://aclanthology.org/2020.acl-main.442/",
    "Beyond Accuracy: Behavioral Testing of NLP Models with CheckList"],
  crowspairs: ["CrowS-Pairs", "https://aclanthology.org/2020.emnlp-main.154/",
    "CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models"],
  fairnessSurvey: ["Bias & Fairness Survey", "https://aclanthology.org/2024.cl-3.8/",
    "Bias and Fairness in Large Language Models: A Survey"],
  robustnessSurvey: ["Robustness Survey", "https://aclanthology.org/2022.naacl-main.339/",
    "Measure and Improve Robustness in NLP Models: A Survey"],
  selfInstruct: ["Self-Instruct", "https://aclanthology.org/2023.acl-long.754/",
    "Self-Instruct: Aligning Language Models with Self-Generated Instructions"],
  caa: ["Steering Llama 2", "https://aclanthology.org/2024.acl-long.828/",
    "Steering Llama 2 via Contrastive Activation Addition"],
  dataShapley: ["Data Shapley", "https://arxiv.org/abs/1904.02868",
    "Data Shapley: Equitable Valuation of Data for Machine Learning"],
  dataAttrSurvey: ["Data Attribution Survey", "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5451054",
    "A Survey of Data Attribution: Methods, Applications, and Evaluation in the Era of Generative AI"],
  datasheets: ["Datasheets for Datasets", "https://arxiv.org/abs/1803.09010",
    "Datasheets for Datasets"],
  dolma: ["Dolma", "https://arxiv.org/abs/2402.00159",
    "Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research"],
  csShapley: ["CS-Shapley", "https://arxiv.org/abs/2211.06800",
    "CS-Shapley: Class-wise Shapley Values for Data Valuation in Classification"],
  consentCrisis: ["Consent in Crisis", "https://proceedings.neurips.cc/paper_files/paper/2024/hash/c3738949a80306cc48a8ea8ba0560f9d-Abstract-Datasets_and_Benchmarks_Track.html",
    "Consent in Crisis: The Rapid Decline of the AI Data Commons"],
  diffAware: ["Difference Awareness", "https://aclanthology.org/2025.acl-long.341/",
    "Fairness through Difference Awareness: Measuring Desired Group Discrimination in LLMs"],
} as const satisfies Record<string, Paper>;

const rawSchedule: ScheduleRow[] = [
  { week: 1, date: "Wed 08/26", topic: "Course Overview, Linguistic Fundamentals, History of NLP", slides: "/slides/wm_fa26_cs680_nlp_lec1_introduction.pdf" },
  { week: 1, date: "Fri 08/28", topic: "Basics of Text Processing", planningTopic: "Basics of Text Processing (Words and Tokens)", materials: readings(jm(2)), slides: "/slides/wm_fa26_cs680_nlp_lec2_textprocessing.pdf" },

  { week: 2, date: "Mon 08/31", topic: "N-Gram Language Models (1)", materials: readings(jm(3)), slides: "/slides/wm_fa26_cs680_nlp_lec3_ngramlms.pdf" },
  { week: 2, date: "Wed 09/02", topic: "N-Gram Language Models (2)", materials: readings(jm(3)), slides: "/slides/wm_fa26_cs680_nlp_lec4_ngramlms.pdf" },
  { week: 2, date: "Fri 09/04", topic: "Text Classification (1)", materials: readings(jm(4)), slides: "/slides/wm_fa26_cs680_nlp_lec5_textclassification1.pdf" },

  { week: 3, date: "Mon 09/07", topic: "No Class - Labor Day" },
  { week: 3, date: "Wed 09/09", topic: "Text Classification (2)", materials: readings(jm(4)), slides: "/slides/wm_fa26_cs680_nlp_lec6_textclassification2.pdf" },
  { week: 3, date: "Fri 09/11", topic: "Word Embeddings (1)", materials: readings(jm(5)), slides: "/slides/wm_fa26_cs680_nlp_lec7_wordembeddings.pdf" },

  { week: 4, date: "Mon 09/14", topic: "Word Embeddings (2)", materials: readings(jm(5)), slides: "/slides/wm_fa26_cs680_nlp_lec8_wordembeddings.pdf" },
  { week: 4, date: "Wed 09/16", topic: "Project Pitches" },
  { week: 4, date: "Fri 09/18", topic: "Project Pitches" },

  { week: 5, date: "Mon 09/21", topic: "Project Pitches & Discussion" },
  { week: 5, date: "Wed 09/23", topic: "Feedforward Networks", materials: readings(jm(6)), slides: "/slides/wm_fa26_cs680_nlp_lec9_ffn.pdf" },
  { week: 5, date: "Fri 09/25", topic: "Backpropagation", materials: readings(jm(6)), slides: "/slides/wm_fa26_cs680_nlp_lec10_backprop.pdf" },

  { week: 6, date: "Mon 09/28", topic: "Hands-On Day" },
  { week: 6, date: "Wed 09/30", topic: "RNNs", materials: readings(jm(14)), slides: "/slides/wm_fa26_cs680_nlp_lec11_rnn.pdf" },
  { week: 6, date: "Fri 10/02", topic: "Seq2Seq & Attention", materials: material(readings(jm(14), jm(7)), papers(P.seq2seq, P.bahdanau)), slides: "/slides/wm_fa26_cs680_nlp_lec12_seq2seq_attn.pdf" },

  { week: 7, date: "Mon 10/05", topic: "Language Generation (1)", materials: material(readings(jm(13), jm(7, "7.6")), papers(P.bleu)), slides: "/slides/wm_fa26_cs680_nlp_lec13_generation.pdf" },
  { week: 7, date: "Wed 10/07", topic: "Language Generation (2)", materials: material(readings(jm(13), jm(7, "7.6")), notebook("/notebooks/csci680_nlp_decoding_strategies.ipynb")), slides: "/slides/wm_fa26_cs680_nlp_lec14_generation.pdf" },
  { week: 7, date: "Fri 10/09", topic: "No Class - Fall Break" },

  { week: 8, date: "Mon 10/12", topic: "Transformers (1)", materials: material(readings(jm(7)), optional(P.attention)), slides: "/slides/wm_fa26_cs680_nlp_lec15_transformers1.pdf" },
  { week: 8, date: "Wed 10/14", topic: "Quiz 2" },
  { week: 8, date: "Fri 10/16", topic: "Transformers (2)", materials: material(readings(jm(7)), optional(P.attention)), slides: "/slides/wm_fa26_cs680_nlp_lec16_transformers2.pdf" },

  { week: 9, date: "Mon 10/19", topic: "Transformer LMs & Pretraining (1)", planningTopic: "Transformer LMs & Pretraining (1): Architectures & Tokenization, BERT & GPT, Pretraining LLMs", materials: material(readings(jm(7), jm(9)), optional(P.bert, P.gpt1), papers(P.bpe, P.nucleus, P.llama3)), slides: "/slides/wm_fa26_cs680_nlp_lec17_pretraining.pdf" },
  { week: 9, date: "Wed 10/21", topic: "Guest Lecture: Scaling Laws & Optimization", materials: material(readings(paper(P.scaling)), handout("Suggested Readings and Repos", "/handouts/wm_fa26_cs680_nlp_scaling_readings_repos.pdf")) },
  { week: 9, date: "Fri 10/23", topic: "Pretraining (2)", materials: material(readings(jm(7), jm(9)), optional(P.bert, P.gpt1), papers(P.bpe, P.nucleus, P.llama3)), slides: "/slides/wm_fa26_cs680_nlp_lec18_pretraining.pdf" },

  { week: 10, date: "Mon 10/26", topic: "Instruction-Tuning", materials: readings(jm(8, "8.1")), slides: "/slides/wm_fa26_cs680_nlp_lec18_posttraining.pdf" },
  { week: 10, date: "Wed 10/28", topic: "Post-Training & Prompting", planningTopic: "Post-Training & Prompting: RLHF, Instruction-Tuning, SFT, DPO, In-Context Learning, Chain-of-Thought", materials: material(readings(jm(8), jm(10)), papers(P.instructgpt, P.dpo, P.r1, P.gpt3, P.cot)), slides: "/slides/wm_fa26_cs680_nlp_lec19_prompting.pdf" },
  { week: 10, date: "Fri 10/30", topic: "Quiz 3" },

  { week: 11, date: "Mon 11/02", topic: "Project Work Day" },
  { week: 11, date: "Wed 11/04", topic: "Evaluating LLMs", materials: material(readings(jm(1, "1.9")), optional(P.checklist, P.mauve, P.helm, P.mmlu)) },
  { week: 11, date: "Fri 11/06", topic: "Paper Discussion 1", materials: readings(paper(P.livebench), paper(P.llmJudge)) },

  { week: 12, date: "Mon 11/09", topic: "Responsible Language Modeling", planningTopic: "Responsible Language Modeling (Harms & Risks)", materials: material(readings(jm(1, "1.10")), optional(P.parrots)) },
  { week: 12, date: "Wed 11/11", topic: "Robustness & Fairness", materials: optional(P.fairnessSurvey, P.robustnessSurvey) },
  { week: 12, date: "Fri 11/13", topic: "Paper Discussion 2", materials: readings(paper(P.consentCrisis), paper(P.diffAware)) },

  { week: 13, date: "Mon 11/16", topic: "Interpretability", materials: material(readings(jm(10)), optional(P.attnNotExpl, P.attnNotNotExpl)) },
  { week: 13, date: "Wed 11/18", topic: "Data-Centric NLP", materials: optional(P.dataAttrSurvey, P.datasheets, P.dolma) },
  { week: 13, date: "Fri 11/20", topic: "Paper Discussion 3", materials: readings(paper(P.selfInstruct), paper(P.caa)) },

  { week: 14, date: "Mon 11/23", topic: "Outro" },
  { week: 14, date: "Wed 11/25", topic: "No Class - Thanksgiving Break" },
  { week: 14, date: "Fri 11/27", topic: "No Class - Thanksgiving Break" },

  { week: 15, date: "Mon 11/30", topic: "Final Project Presentations" },
  { week: 15, date: "Wed 12/02", topic: "Final Project Presentations" },
  { week: 15, date: "Fri 12/04", topic: "Final Project Presentations" },
];

/**
 * Materials are revealed to students week by week. Readings and papers are
 * shown on /nlp up to and including this session; everything after it is
 * hidden there but stays visible on /nlp/planning.
 *
 * Move this date forward as the semester progresses. Set it to "" to show
 * everything, or to the first session to hide everything.
 */
export const materialsVisibleThrough = "Fri 11/20";

/**
 * Suggested papers are revealed on their own schedule, always at or behind
 * `materialsVisibleThrough`: a session past this date shows its readings on
 * /nlp but not its papers. They stay visible on /nlp/planning either way.
 *
 * Same convention as above — "" shows every paper list, the first session
 * hides them all.
 */
export const papersVisibleThrough = "Wed 08/26";

const cutoff = rawSchedule.findIndex((r) => r.date === materialsVisibleThrough);
const papersCutoff = rawSchedule.findIndex((r) => r.date === papersVisibleThrough);

export const schedule: ScheduleRow[] = rawSchedule.map((row, i) => {
  if (!row.materials) return row;
  return {
    ...row,
    ...(cutoff !== -1 && i > cutoff && { materialsHidden: true }),
    ...(papersCutoff !== -1 &&
      i > papersCutoff &&
      row.materials.includes(PAPERS_LABEL) && { papersHidden: true }),
  };
});

export interface RemovedItem {
  topic: string;
  materials?: string;
  note?: string;
}

// Parked topics, kept so their readings and papers are not lost. Rendered only
// on /nlp/planning, which is stripped from the published site.
export const removedContent: RemovedItem[] = [];
