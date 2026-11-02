# AgentForStudent

مشروع مادة AI3101: وكيل ذكي يساعد الطالب الجامعي في دراسته. يتتبّع مستوى فهم الطالب لكل مجال (BKT)، ويتوقع المواد اللي ممكن يتعثر فيها (Random Forest)، ويسوي له جدول مذاكرة ومواعيد مراجعة (CSP + SM-2)، ويجاوب أسئلته من مادة المقرر (RAG).

## الملفات
| الملف | وش هو |
| --- | --- |
| `students_by_term_150.xlsx` | الداتا: 300 طالب (محاكاة)، و7,912 درجة، و43 مادة |
| `AgentForStudent.ipynb` | الكود بصيغة دفتر Colab |
| `agent_for_student.py` | نفس الكود كملف بايثون واحد (لـ VS Code) |

## طريقة التشغيل
**في Google Colab:** افتح [الكود في Colab](https://colab.research.google.com/github/T5o2/AgentForStudent/blob/main/AgentForStudent.ipynb)، واضغط **Runtime ← Run all**. الكود يحمّل الداتا بنفسه.

**في VS Code:** حط ملف الإكسل جنب ملف الكود، وشغّل الأوامر هذي:
```
pip install pandas scikit-learn matplotlib openpyxl
python agent_for_student.py students_by_term_150.xlsx out
```

## الفريق
- تركي محمد تركي الدوسري
- عيد وليد عيد الجهني
- عبدالمجيد محمد بخروش الزهراني
