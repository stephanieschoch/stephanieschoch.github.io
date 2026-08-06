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
  institution: "William & Mary",
  semester: "Fall 2026",
  time: "TBD",
  location: "TBD",
  description:
    "This course provides an introduction to natural language processing (NLP) with a focus on modern methods and large language models. Topics include text classification, language modeling, sequence-to-sequence models, attention mechanisms, transformers, pre-training, fine-tuning, prompting, and evaluation. Students will gain both theoretical understanding and practical experience through assignments and a final project.",
  instructors: [
    { name: "Stephanie Schoch", url: "/" },
  ] as Instructor[],
  officeHours: "TBD",
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

export const schedule: ScheduleRow[] = [
  { week: 1, date: "Thu 08/27", topic: "Course Overview, Linguistic Fundamentals, History of NLP" },
  { week: 2, date: "Tue 09/01", topic: "Basics of Text Processing (Words and Tokens)", materials: "Optional Readings:\n- [J&M ch. 2](https://web.stanford.edu/~jurafsky/slp3/2.pdf)" },
  { week: 2, date: "Thu 09/03", topic: "N-Gram Language Models", materials: "Optional Readings:\n- [J&M ch. 3](https://web.stanford.edu/~jurafsky/slp3/3.pdf)" },
  { week: 3, date: "Tue 09/08", topic: "Text Classification", materials: "Optional Readings:\n- [J&M ch. 4](https://web.stanford.edu/~jurafsky/slp3/4.pdf)" },
  { week: 3, date: "Thu 09/10", topic: "Word Embeddings", materials: "Optional Readings:\n- [J&M ch. 5](https://web.stanford.edu/~jurafsky/slp3/5.pdf)" },
  { week: 4, date: "Tue 09/15", topic: "Project Pitches" },
  { week: 4, date: "Thu 09/17", topic: "Feedforward Networks", materials: "Optional Readings:\n- [J&M ch. 6](https://web.stanford.edu/~jurafsky/slp3/6.pdf)" },
  { week: 5, date: "Tue 09/22", topic: "Backpropogation and RNNs", materials: "Optional Readings:\n- [J&M ch. 13](https://web.stanford.edu/~jurafsky/slp3/13.pdf)" },
  { week: 5, date: "Thu 09/24", topic: "Seq2Seq & Attention" },
  { week: 6, date: "Tue 09/29", topic: "Transformers", materials: "Optional Readings:\n- [J&M ch. 8](https://web.stanford.edu/~jurafsky/slp3/8.pdf)" },
  { week: 6, date: "Thu 10/01", topic: "Transformer LMs (BERT, GPT)", materials: "Optional Readings:\n- [J&M ch. 7](https://web.stanford.edu/~jurafsky/slp3/7.pdf)\n- [J&M ch. 10](https://web.stanford.edu/~jurafsky/slp3/10.pdf)" },
  { week: 7, date: "Tue 10/06", topic: "Tokenization in Modern LMs; Scaling Laws" },
  { week: 7, date: "Thu 10/08", topic: "No Class - Fall Break" },
  { week: 8, date: "Tue 10/13", topic: "NLG (e.g. Decoding Strategies)" },
  { week: 8, date: "Thu 10/15", topic: "Pre-Training LLMs", materials: "Optional Readings:\n- [J&M ch. 7](https://web.stanford.edu/~jurafsky/slp3/7.pdf)" },
  { week: 9, date: "Tue 10/20", topic: "Post-Training: (RLHF, Instruction-Tuning, SFT, DPO)", materials: "Optional Readings:\n- [J&M ch. 9](https://web.stanford.edu/~jurafsky/slp3/9.pdf)" },
  { week: 9, date: "Thu 10/22", topic: "Fine-Tuning, Transfer Learning, Efficient Adaptation (PEFT, LORA)" },
  { week: 10, date: "Tue 10/27", topic: "Prompting & ICL" },
  { week: 10, date: "Thu 10/29", topic: "Reasoning" },
  { week: 11, date: "Tue 11/03", topic: "No Class - Election Day" },
  { week: 11, date: "Thu 11/05", topic: "Evaluating LLMs" },
  { week: 12, date: "Tue 11/10", topic: "Responsible Language Modeling (Harms & Risks) / Interpretability, Robustness, Fairness" },
  { week: 12, date: "Thu 11/12", topic: "Paper Discussion" },
  { week: 13, date: "Tue 11/17", topic: "Data-Centric NLP" },
  { week: 13, date: "Thu 11/19", topic: "Paper Discussion" },
  { week: 14, date: "Tue 11/24", topic: "Hands-On Day/Outro/Project Work Day" },
  { week: 14, date: "Thu 11/26", topic: "No Class - Thanksgiving Break" },
  { week: 15, date: "Tue 12/01", topic: "Final Project Presentations" },
  { week: 15, date: "Thu 12/03", topic: "Final Project Presentations" },
];
