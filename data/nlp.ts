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
}

export const courseInfo = {
  title: "Natural Language Processing",
  number: "CSCI 680",
  section: "04",
  institution: "William & Mary",
  semester: "Fall 2026",
  time: "MWF 10:00–10:50am",
  location: "Integrated Science Center (ISC) 3280",
  description:
    "This course provides an introduction to natural language processing (NLP) with a focus on modern methods and large language models. Topics include text classification, language modeling, sequence-to-sequence models, attention mechanisms, transformers, pre-training, fine-tuning, prompting, and evaluation. Students will gain both theoretical understanding and practical experience through assignments and a final project.",
  instructors: [
    { name: "Stephanie Schoch", url: "/" },
  ] as Instructor[],
  officeHours: "M 1:00 - 2:00 PM",
  prerequisites: "Students should be proficient in Python. Experience with packages such as SciPy, Scikit-learn, and PyTorch is helpful. Students should also have experience with Calculus, Linear Algebra, and Probability & Statistics.",
};

export interface DeadlineRow {
  week: number;
  deadline: string;
  date: string;
  time?: string;
}

export const deadlines: DeadlineRow[] = [
  { week: 1, deadline: "TBD", date: "TBD", time: "" },
];

const jm = (ch: number | string) =>
  `[J&M ch. ${ch}](https://web.stanford.edu/~jurafsky/slp3/${ch}.pdf)`;
const readings = (...chapters: (number | string)[]) =>
  `Optional Readings:\n${chapters.map((c) => `- ${jm(c)}`).join("\n")}`;

export const schedule: ScheduleRow[] = [
  { week: 1, date: "Wed 08/26", topic: "Course Overview, Linguistic Fundamentals, History of NLP" },
  { week: 1, date: "Fri 08/28", topic: "Basics of Text Processing (Words and Tokens)", materials: readings(2) },

  { week: 2, date: "Mon 08/31", topic: "N-Gram Language Models", materials: readings(3) },
  { week: 2, date: "Wed 09/02", topic: "Text Classification (1)", materials: readings(4) },
  { week: 2, date: "Fri 09/04", topic: "Text Classification (2)" },

  { week: 3, date: "Mon 09/07", topic: "No Class - Labor Day" },
  { week: 3, date: "Wed 09/09", topic: "Word Embeddings (1)", materials: readings(5) },
  { week: 3, date: "Fri 09/11", topic: "Word Embeddings (2)" },

  { week: 4, date: "Mon 09/14", topic: "Hands-On Day" },
  { week: 4, date: "Wed 09/16", topic: "Project Pitches" },
  { week: 4, date: "Fri 09/18", topic: "Project Pitches" },

  { week: 5, date: "Mon 09/21", topic: "Feedforward Networks", materials: readings(6) },
  { week: 5, date: "Wed 09/23", topic: "Backpropagation and RNNs", materials: readings(13) },
  { week: 5, date: "Fri 09/25", topic: "RNNs (Cont.)" },

  { week: 6, date: "Mon 09/28", topic: "Seq2Seq" },
  { week: 6, date: "Wed 09/30", topic: "Attention" },
  { week: 6, date: "Fri 10/02", topic: "Transformers (1)", materials: readings(8) },

  { week: 7, date: "Mon 10/05", topic: "Transformers (2)" },
  { week: 7, date: "Wed 10/07", topic: "Transformer LMs (BERT, GPT)", materials: readings(7, 10) },
  { week: 7, date: "Fri 10/09", topic: "No Class - Fall Break" },

  { week: 8, date: "Mon 10/12", topic: "Tokenization in Modern LMs; Scaling Laws" },
  { week: 8, date: "Wed 10/14", topic: "NLG (Decoding Strategies)" },
  { week: 8, date: "Fri 10/16", topic: "NLG (Cont.)" },

  { week: 9, date: "Mon 10/19", topic: "Pre-Training LLMs", materials: readings(7) },
  { week: 9, date: "Wed 10/21", topic: "Post-Training: (RLHF, Instruction-Tuning, SFT, DPO)", materials: readings(9) },
  { week: 9, date: "Fri 10/23", topic: "Fine-Tuning, Transfer Learning, Efficient Adaptation (PEFT, LORA)" },

  { week: 10, date: "Mon 10/26", topic: "Prompting & ICL" },
  { week: 10, date: "Wed 10/28", topic: "Reasoning" },
  { week: 10, date: "Fri 10/30", topic: "Hands-On Day" },

  { week: 11, date: "Mon 11/02", topic: "Evaluating LLMs" },
  { week: 11, date: "Wed 11/04", topic: "Agents, Tool Use, RAG" },
  { week: 11, date: "Fri 11/06", topic: "Paper Discussion" },

  { week: 12, date: "Mon 11/09", topic: "Responsible Language Modeling (Harms & Risks)" },
  { week: 12, date: "Wed 11/11", topic: "Interpretability, Robustness, Fairness" },
  { week: 12, date: "Fri 11/13", topic: "Paper Discussion" },

  { week: 13, date: "Mon 11/16", topic: "Data-Centric NLP (1)" },
  { week: 13, date: "Wed 11/18", topic: "Data-Centric NLP (2)" },
  { week: 13, date: "Fri 11/20", topic: "Paper Discussion" },

  { week: 14, date: "Mon 11/23", topic: "Outro/Project Work Day" },
  { week: 14, date: "Wed 11/25", topic: "No Class - Thanksgiving Break" },
  { week: 14, date: "Fri 11/27", topic: "No Class - Thanksgiving Break" },

  { week: 15, date: "Mon 11/30", topic: "Final Project Presentations" },
  { week: 15, date: "Wed 12/02", topic: "Final Project Presentations" },
  { week: 15, date: "Fri 12/04", topic: "Final Project Presentations" },
];
