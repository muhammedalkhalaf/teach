# دليل حل المشاكل - ARDL Shiny App

## مشاكل التثبيت

### 1. فشل تثبيت حزمة ARDL

**الأعراض:**
```
Error: package 'ARDL' is not available
```

**الحل:**
```r
# تحديث CRAN
options(repos = c(CRAN = "https://cran.r-project.org"))

# إعادة المحاولة
install.packages("ARDL", dependencies = TRUE)

# إذا استمرت المشكلة، جرّب من GitHub
install.packages("devtools")
devtools::install_github("Natsiopoulos/ARDL")
```

### 2. خطأ في تثبيت urca

**الأعراض:**
```
Error: compilation failed for package 'urca'
```

**الحل (Linux):**
```bash
# تثبيت المكتبات المطلوبة
sudo apt-get update
sudo apt-get install libgfortran5 gfortran
```

**الحل (Mac):**
```bash
brew install gcc
```

**الحل (Windows):**
- تثبيت Rtools من: https://cran.r-project.org/bin/windows/Rtools/

### 3. خطأ في تثبيت officer

**الأعراض:**
```
Error: package 'officer' had non-zero exit status
```

**الحل:**
```r
# تثبيت الاعتماديات أولاً
install.packages(c("xml2", "zip", "gdtools"))

# ثم تثبيت officer
install.packages("officer")
```

---

## مشاكل التشغيل

### 4. التطبيق لا يفتح

**الأعراض:**
```
Error in shiny::runApp("app.R") :
  could not find function "runApp"
```

**الحل:**
```r
# تحميل shiny أولاً
library(shiny)
runApp("app.R")
```

### 5. خطأ في تحميل helpers.R

**الأعراض:**
```
Error in source("helpers.R") : cannot open the connection
```

**الحل:**
```r
# تأكد من المجلد الصحيح
getwd()  # يجب أن يعرض مسار ardl_shiny_app

# إذا لم يكن كذلك:
setwd("path/to/ardl_shiny_app")
```

### 6. الواجهة العربية لا تظهر بشكل صحيح

**الأعراض:**
- النصوص العربية تظهر كرموز غريبة
- التخطيط من اليسار لليمين بدلاً من اليمين لليسار

**الحل:**
```r
# تأكد من encoding UTF-8
Sys.setlocale("LC_ALL", "en_US.UTF-8")

# أو في Windows:
Sys.setlocale("LC_ALL", "Arabic_Saudi Arabia.1256")
```

---

## مشاكل البيانات

### 7. خطأ عند تحميل ملف CSV

**الأعراض:**
```
Error: 'data_raw' is NULL
```

**الحلول:**

**أ) ترميز الملف:**
```r
# حاول تحديد الترميز يدوياً
data <- read.csv("file.csv", encoding = "UTF-8")
```

**ب) الفواصل:**
```r
# إذا كان الملف يستخدم ; بدلاً من ,
data <- read.csv("file.csv", sep = ";")

# أو استخدم read_csv من readr (أذكى)
library(readr)
data <- read_csv("file.csv")
```

**ج) الأسطر الفارغة:**
- افتح الملف في Excel/LibreOffice
- احذف أي أسطر فارغة في البداية أو النهاية
- احفظ مجدداً

### 8. خطأ: "المتغيرات غير رقمية"

**الأعراض:**
```
Error: all variables must be numeric
```

**الحل:**
```r
# تحقق من نوع الأعمدة
str(data)

# تحويل إلى رقمي إذا لزم
data$Variable <- as.numeric(data$Variable)

# إزالة الفواصل أو الرموز
data$GDP <- as.numeric(gsub(",", "", data$GDP))
```

### 9. عمود التاريخ لا يُتعرف عليه

**الأعراض:**
- الفهرس الزمني غير صحيح

**الحل:**
```r
# تحويل إلى صيغة التاريخ أولاً
data$Date <- as.Date(data$Date, format = "%Y-%m-%d")

# للصيغ الأخرى:
# "%d/%m/%Y" لـ 31/12/2020
# "%m/%d/%Y" لـ 12/31/2020
# "%Y" لـ 2020 فقط
```

---

## مشاكل التقدير

### 10. "فشل تقدير النموذج" لجميع النماذج

**الأسباب المحتملة:**

**أ) بيانات قليلة جداً:**
```
الحل: استخدم 40+ مشاهدة على الأقل
```

**ب) متغيرات ثابتة:**
```r
# تحقق من التباين
sapply(data, var, na.rm = TRUE)

# إذا كان التباين = 0 أو قريب جداً منه، أزل المتغير
```

**ج) قيم متطرفة:**
```r
# فحص القيم المتطرفة
boxplot(data$Variable)

# معالجتها (winsorizing)
library(DescTools)
data$Variable <- Winsorize(data$Variable, probs = c(0.01, 0.99))
```

**د) تأخيرات كبيرة جداً:**
```
الحل: قلل max_p و max_q إلى 2 أو 3
```

### 11. لا توجد نماذج ناجحة (كلها مستبعدة)

**الأعراض:**
```
"لم يجتاز أي نموذج المرشّحات الصارمة"
```

**الحلول:**

**أ) الارتباط الذاتي في جميع النماذج:**
```
→ زد max_p إلى 6 أو 8
→ تحقق من وجود تغييرات هيكلية
→ أضف متغيرات افتراضية للأزمات
```

**ب) ECT موجب في جميع النماذج:**
```
→ قد لا يوجد تكامل مشترك
→ جرّب VAR بدلاً من ARDL
→ تحقق من استقرارية المتغيرات
```

**ج) عدم استقرار شديد:**
```
→ تحقق من البيانات للقيم المتطرفة
→ جرّب نطاقات تأخير مختلفة
→ استخدم حتميات مختلفة
```

### 12. Bounds Test يفشل

**الأعراض:**
```
bounds$failed = TRUE
```

**الحل:**
```r
# تأكد من case number الصحيح
# 1 = no constant, no trend
# 2 = unrestricted constant, no trend
# 3 = restricted constant, no trend
# 4 = unrestricted constant, unrestricted trend
# 5 = unrestricted constant, restricted trend

# جرّب cases مختلفة يدوياً
```

---

## مشاكل الأداء

### 13. التطبيق بطيء جداً (ساعات!)

**الأسباب:**

**أ) عدد النماذج كبير جداً:**
```
عدد_النماذج = (max_p + 1) × (max_q + 1)^n_vars × n_deterministics

مثال: 5 × 5³ × 4 = 2500 نموذج!
```

**الحلول:**
```r
# 1. قلل الحدود
max_p = 2
max_q = 2
# النتيجة: 3 × 3² × 4 = 108 نموذج (أسرع بـ 23x!)

# 2. اختر حتمية محددة
deterministics = "const"  # بدلاً من "auto"

# 3. فعّل المعالجة المتوازية
parallel_processing = TRUE
n_cores = 4  # حسب معالجك
```

### 14. الذاكرة ممتلئة (Memory Error)

**الأعراض:**
```
Error: cannot allocate vector of size X Gb
```

**الحل:**
```r
# 1. زد حد الذاكرة (Windows)
memory.limit(size = 16000)  # 16 GB

# 2. قلل عدد النماذج (انظر #13)

# 3. استخدم standalone script بدلاً من Shiny
source("standalone_reestimate.R")

# 4. معالجة دفعية
# قسّم التحليل إلى دفعات صغيرة
```

### 15. التطبيق يتجمد (Freeze)

**الأعراض:**
- شريط التقدم يتوقف
- لا استجابة

**الحل:**
```r
# 1. تحقق من وحدة التحكم للأخطاء

# 2. أضف print statements للتتبع
# في helpers.R:
cat("Processing model", i, "of", n_models, "\n")

# 3. اختبر خارج Shiny أولاً
# اختبر fit_ardl_candidate() يدوياً

# 4. حدد timeout أطول
options(timeout = 300)  # 5 دقائق
```

---

## مشاكل الرسوم

### 16. الرسوم لا تظهر

**الأعراض:**
- مساحة فارغة بدلاً من الرسم

**الحل:**
```r
# 1. تحقق من المتصفح
# يُنصح بـ Chrome أو Firefox

# 2. تحديث الصفحة
# Ctrl+F5 (Windows) أو Cmd+Shift+R (Mac)

# 3. تحقق من الأخطاء في console
# F12 → Console tab

# 4. جرّب حفظ الرسم مباشرة
pdf("test_plot.pdf")
plot(data$Variable, type = "l")
dev.off()
```

### 17. رسوم ACF/PACF غير واضحة

**الحل:**
```r
# زد الدقة
plotOutput("acf_plots", height = "1000px")

# أو حمّل كـ PDF للجودة العالية
```

### 18. النصوص العربية في الرسوم تظهر كمربعات

**الحل:**
```r
# تثبيت خطوط عربية
# Linux:
system("sudo apt-get install fonts-arabic")

# تحديد الخط في ggplot
library(ggplot2)
theme_set(theme_minimal(base_family = "DejaVu Sans"))

# أو استخدم Cairo
library(Cairo)
Cairo::CairoFonts(
  regular = "DejaVu Sans:style=Book"
)
```

---

## مشاكل التصدير

### 19. فشل تحميل CSV

**الأعراض:**
```
Error in download handler
```

**الحل:**
```r
# تأكد من وجود النتائج أولاً
req(rv$ranked_models)

# تحقق من الترميز
write.csv(data, file, fileEncoding = "UTF-8", row.names = FALSE)
```

### 20. تقرير Word فارغ أو مشوّه

**الأعراض:**
- ملف Word يفتح لكنه فارغ
- التنسيق غير صحيح

**الحل:**
```r
# 1. تحديث officer و flextable
install.packages(c("officer", "flextable"))

# 2. استخدم قالب بسيط
# تجنب التنسيقات المعقدة في الإصدار الأول

# 3. اختبر خارج Shiny
library(officer)
doc <- read_docx()
doc <- body_add_par(doc, "اختبار عربي")
print(doc, target = "test.docx")
```

---

## مشاكل الاختبارات التشخيصية

### 21. Breusch-Godfrey يفشل دائماً

**الأعراض:**
- جميع النماذج لديها ارتباط ذاتي

**الحلول:**
```r
# 1. زد رتبة التأخيرات
max_p = 6

# 2. غيّر رتبة BG test
bg_order = 4  # بدلاً من 2

# 3. تحقق من وجود outliers
# 4. أضف متغيرات افتراضية
```

### 22. CUSUM/CUSUMSQ يظهران عدم استقرار

**الأعراض:**
- الخط الأزرق يخرج عن الحدود الحمراء

**التفسير:**
```
→ يوجد تغيير هيكلي (structural break)
→ قد يكون بسبب أزمة أو تغيير سياسة
```

**الحل:**
```r
# 1. حدد نقطة التغيير
breakpoints(model)

# 2. أضف متغير افتراضي
data$D_crisis <- ifelse(data$year >= 2008, 1, 0)

# 3. أعد التقدير مع الـ dummy
```

### 23. Jarque-Bera يفشل (البواقي غير طبيعية)

**التفسير:**
```
→ توزيع البواقي غير طبيعي
→ قد لا يكون مشكلة كبيرة في عينات كبيرة
```

**الحل:**
```r
# 1. تحقق من القيم المتطرفة
boxplot(residuals(model))

# 2. استخدم robust standard errors
library(sandwich)
coeftest(model, vcov = vcovHC)

# 3. زد حجم العينة إن أمكن
```

---

## مشاكل التفسير

### 24. كيف أعرف إذا كانت النتائج جيدة؟

**قائمة التحقق:**

✅ **تشخيصات ناجحة:**
- [ ] BG: p > 0.05 (لا ارتباط ذاتي)
- [ ] ECT: سالب ومعنوي
- [ ] CUSUM/CUSUMSQ: داخل الحدود
- [ ] Bounds Test: F > Upper bound

✅ **معاملات منطقية:**
- [ ] الإشارات صحيحة اقتصادياً
- [ ] الحجم معقول (ليس 0.00001 ولا 10000)

✅ **جودة الملاءمة:**
- [ ] R² معقول (> 0.7 للسلاسل الزمنية عادة)
- [ ] القيم الملائمة قريبة من الفعلية

### 25. معامل ECT قريب من صفر (-0.01)

**التفسير:**
```
→ التعديل نحو التوازن بطيء جداً
→ قد يستغرق 100 فترة للعودة للتوازن
```

**هل هذا سيء؟**
```
ليس بالضرورة، لكن:
- إذا كان معنوياً: لا بأس
- إذا كان غير معنوي: قد لا يوجد تكامل مشترك حقيقي
```

### 26. AIC و BIC يختلفان في الترتيب

**التفسير:**
```
→ BIC يعاقب النماذج المعقدة أكثر
→ AIC يفضل النماذج الأكثر ملاءمة
```

**ما العمل؟**
```
هذا التطبيق يستخدم:
1. التشخيصات أولاً (الأهم!)
2. النقاط المركبة
3. AIC/BIC كمرجع فقط

→ اتبع النموذج الأفضل حسب النقاط الكلية
```

---

## مشاكل أخرى

### 27. التطبيق يُغلق فجأة

**الأسباب المحتملة:**
```r
# 1. خطأ في الكود
# → تحقق من console للأخطاء

# 2. ذاكرة ممتلئة
# → قلل عدد النماذج

# 3. timeout
# → زد الحد
options(shiny.maxRequestSize = 50*1024^2)  # 50 MB
```

### 28. لا أستطيع رفع ملف كبير

**الحل:**
```r
# زد حد حجم الملف
options(shiny.maxRequestSize = 100*1024^2)  # 100 MB

# ضع هذا في app.R قبل shinyApp()
```

### 29. كيف أحفظ جلستي؟

**الحل:**
```r
# 1. استخدم السكربت المستقل
source("standalone_reestimate.R")

# 2. احفظ workspace
save.image("my_analysis.RData")

# 3. لإعادة التحميل
load("my_analysis.RData")

# 4. احفظ النموذج فقط
saveRDS(best_model, "model.rds")
model <- readRDS("model.rds")
```

---

## الحصول على المساعدة

إذا لم تجد الحل هنا:

### 1. تحقق من sessionInfo
```r
sessionInfo()

# أرسل المخرجات عند طلب المساعدة
```

### 2. أنشئ مثال قابل للإعادة
```r
# استخدم البيانات التجريبية
data <- generate_synthetic_data(50)

# أعد إنتاج المشكلة
# ...
```

### 3. ابحث في الوثائق
- README.md
- QUICKSTART.md
- ?function_name

### 4. تحقق من رسائل الخطأ بعناية
```
غالباً ما تحتوي على الحل!
```

---

**تم تحديث:** 2025-10-22
**الإصدار:** 1.0.0
