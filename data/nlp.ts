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
  { week: 5, deadline: "Homework 1", released: "Wed 09/09", date: "Fri 09/25", time: "11:59 PM" },
  { week: 6, deadline: "Project Proposal", date: "Fri 10/02", time: "11:59 PM" },
  { week: 8, deadline: "Quiz 2: Neural Methods", date: "Mon 10/12", time: "10:00 AM" },
  { week: 9, deadline: "Homework 2", released: "Mon 09/28", date: "Fri 10/23", time: "11:59 PM" },
  { week: 10, deadline: "Quiz 3: LLMs", date: "Fri 10/30", time: "10:00 AM" },
  { week: 11, deadline: "Project Progress Report", date: "Mon 11/02", time: "11:59 PM" },
  { week: 13, deadline: "Homework 3", released: "Mon 10/26", date: "Fri 11/20", time: "11:59 PM" },
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
// Section numbers were read from the Jan 6 2026 chapter PDFs — this is a living
// draft and sections do move between releases, so re-verify each semester.
const SLP = "https://web.stanford.edu/~jurafsky/slp3";
// Note: ch. 1 has no standalone PDF (slp3/1.pdf is a 404); it exists only inside
// the ~25MB full-text book, so it is not cited here.
// Rendered as "J&M: §7.4", or "J&M: §3.4, §12.4" when a day has several.
const jm = (ch: number, sec?: string) => `[§${sec ?? ch}](${SLP}/${ch}.pdf)`;

// The "J&M:" prefix goes *inside* the first link so the whole reference is
// clickable — otherwise the prefix renders in body colour beside accent-coloured
// links and reads like a dead link.
const readings = (...refs: string[]) => {
  const [first, ...rest] = refs;
  return `Readings: ${[first.replace(/^\[/, "[J&M: "), ...rest].join(", ")}`;
};

// [short display name, url, full title]. The short name is what appears in the
// schedule; the full title becomes the link's hover tooltip.
type Paper = readonly [short: string, url: string, full: string];
const papers = (...ps: Paper[]) =>
  `Suggested Papers: ${ps.map(([s, u, f]) => `[${s}](${u} "${f}")`).join(", ")}`;

// The renderer turns each run of "- " lines into its own <ul>, so the two
// groups stay visually separate. See app/nlp/page.tsx.
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
  checklist: ["CheckList", "https://aclanthology.org/2020.acl-main.442/",
    "Beyond Accuracy: Behavioral Testing of NLP Models with CheckList"],
  crowspairs: ["CrowS-Pairs", "https://aclanthology.org/2020.emnlp-main.154/",
    "CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models"],
  dataShapley: ["Data Shapley", "https://arxiv.org/abs/1904.02868",
    "Data Shapley: Equitable Valuation of Data for Machine Learning"],
  datasheets: ["Datasheets for Datasets", "https://arxiv.org/abs/1803.09010",
    "Datasheets for Datasets"],
  dolma: ["Dolma", "https://arxiv.org/abs/2402.00159",
    "Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research"],
  csShapley: ["CS-Shapley", "https://arxiv.org/abs/2211.06800",
    "CS-Shapley: Class-wise Shapley Values for Data Valuation in Classification"],
} as const satisfies Record<string, Paper>;

const rawSchedule: ScheduleRow[] = [
  { week: 1, date: "Wed 08/26", topic: "Course Overview, Linguistic Fundamentals, History of NLP" },
  { week: 1, date: "Fri 08/28", topic: "Basics of Text Processing", planningTopic: "Basics of Text Processing (Words and Tokens)", materials: readings(jm(2)) },

  { week: 2, date: "Mon 08/31", topic: "N-Gram Language Models (1)", materials: readings(jm(3)) },
  { week: 2, date: "Wed 09/02", topic: "N-Gram Language Models (2)", materials: readings(jm(3)) },
  { week: 2, date: "Fri 09/04", topic: "Text Classification (1)", materials: readings(jm(4)) },

  { week: 3, date: "Mon 09/07", topic: "No Class - Labor Day" },
  { week: 3, date: "Wed 09/09", topic: "Text Classification (2)", materials: readings(jm(4)) },
  { week: 3, date: "Fri 09/11", topic: "Word Embeddings (1)", materials: material(readings(jm(5)), papers(P.word2vec)) },

  { week: 4, date: "Mon 09/14", topic: "Word Embeddings (2)", materials: material(readings(jm(5)), papers(P.glove)) },
  { week: 4, date: "Wed 09/16", topic: "Project Pitches" },
  { week: 4, date: "Fri 09/18", topic: "Project Pitches" },

  { week: 5, date: "Mon 09/21", topic: "Feedforward Networks", materials: readings(jm(6)) },
  { week: 5, date: "Wed 09/23", topic: "Backpropagation and RNNs", materials: readings(jm(13)) },
  { week: 5, date: "Fri 09/25", topic: "RNNs (Cont.)", materials: readings(jm(13, "13.4–13.5")) },

  { week: 6, date: "Mon 09/28", topic: "Seq2Seq", materials: material(readings(jm(13, "13.7")), papers(P.seq2seq)) },
  { week: 6, date: "Wed 09/30", topic: "Attention", materials: material(readings(jm(8, "8.1")), papers(P.bahdanau)) },
  { week: 6, date: "Fri 10/02", topic: "Language Generation", materials: material(readings(jm(3, "3.4"), jm(12, "12.4"), jm(12, "12.6")), papers(P.bleu)) },

  { week: 7, date: "Mon 10/05", topic: "Transformers (1)", materials: material(readings(jm(8, "8.2–8.3")), papers(P.attention)) },
  { week: 7, date: "Wed 10/07", topic: "Transformers (2)", materials: readings(jm(8, "8.4–8.5")) },
  { week: 7, date: "Fri 10/09", topic: "No Class - Fall Break" },

  { week: 8, date: "Mon 10/12", topic: "Hands-On Day" },
  { week: 8, date: "Wed 10/14", topic: "Transformer LMs (1)", planningTopic: "Transformer LMs (1): Architectures & Tokenization", materials: material(readings(jm(7), jm(2, "2.4")), papers(P.bpe)) },
  { week: 8, date: "Fri 10/16", topic: "Transformer LMs (2)", planningTopic: "Transformer LMs (2): BERT & GPT", materials: material(readings(jm(10), jm(8, "8.6")), papers(P.bert, P.gpt1, P.nucleus)) },

  { week: 9, date: "Mon 10/19", topic: "Pre-Training LLMs", materials: material(readings(jm(7, "7.5")), papers(P.llama3)) },
  { week: 9, date: "Wed 10/21", topic: "Guest Lecture: Scaling Laws & Optimization", materials: material(readings(jm(8, "8.7"), jm(8, "8.8.1")), papers(P.scaling)) },
  { week: 9, date: "Fri 10/23", topic: "Post-Training", planningTopic: "Post-Training: (RLHF, Instruction-Tuning, SFT, DPO)", materials: material(readings(jm(9, "9.1–9.3")), papers(P.instructgpt, P.dpo, P.r1)) },

  { week: 10, date: "Mon 10/26", topic: "Fine-Tuning, Efficient Adaptation", planningTopic: "Fine-Tuning, Transfer Learning, Efficient Adaptation (PEFT, LORA)", materials: material(readings(jm(8, "8.8.3")), papers(P.lora, P.adapters)) },
  { week: 10, date: "Wed 10/28", topic: "Prompting, In-Context Learning, and Chain-of-Thought", materials: material(readings(jm(7, "7.3"), jm(8, "8.9.1"), jm(9, "9.4")), papers(P.gpt3, P.cot)) },
  { week: 10, date: "Fri 10/30", topic: "LLM Agents", planningTopic: "LLM Agents, Tool Use, RAG", materials: material(readings(jm(11)), papers(P.rag, P.react, P.toolformer)) },

  { week: 11, date: "Mon 11/02", topic: "Evaluating LLMs", materials: material(readings(jm(7, "7.6")), papers(P.helm, P.mmlu, P.mauve)) },
  { week: 11, date: "Wed 11/04", topic: "Responsible Language Modeling", planningTopic: "Responsible Language Modeling (Harms & Risks)", materials: material(readings(jm(7, "7.7")), papers(P.parrots, P.risks)) },
  { week: 11, date: "Fri 11/06", topic: "Paper Discussion 1" },

  { week: 12, date: "Mon 11/09", topic: "Project Work Day" },
  { week: 12, date: "Wed 11/11", topic: "Interpretability", materials: material(readings(jm(8, "8.9")), papers(P.attnNotExpl)) },
  { week: 12, date: "Fri 11/13", topic: "Paper Discussion 2" },

  { week: 13, date: "Mon 11/16", topic: "Robustness & Fairness", materials: papers(P.checklist, P.crowspairs) },
  { week: 13, date: "Wed 11/18", topic: "Data-Centric NLP", materials: papers(P.dataShapley, P.dolma) },
  { week: 13, date: "Fri 11/20", topic: "Paper Discussion 3" },

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
export const materialsVisibleThrough = "Wed 09/09";

const cutoff = rawSchedule.findIndex((r) => r.date === materialsVisibleThrough);

export const schedule: ScheduleRow[] = rawSchedule.map((row, i) =>
  cutoff !== -1 && i > cutoff && row.materials
    ? { ...row, materialsHidden: true }
    : row
);

export interface RemovedItem {
  topic: string;
  materials?: string;
  note?: string;
}

// Parked topics, kept so their readings and papers are not lost. Rendered only
// on /nlp/planning, which is stripped from the published site.
export const removedContent: RemovedItem[] = [];
