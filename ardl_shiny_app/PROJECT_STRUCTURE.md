# بنية المشروع - ARDL Shiny App

## نظرة عامة

```
ardl_shiny_app/
│
├── app.R                      # التطبيق الرئيسي (UI + Server)
├── helpers.R                  # الدوال المساعدة
├── install.R                  # سكربت تثبيت الحزم
├── standalone_reestimate.R    # سكربت مستقل لإعادة التقدير
│
├── README.md                  # دليل المستخدم الكامل
├── QUICKSTART.md              # دليل البدء السريع
├── TROUBLESHOOTING.md         # حل المشاكل الشائعة
├── PROJECT_STRUCTURE.md       # هذا الملف
├── DESCRIPTION                # وصف الحزمة
├── LICENSE                    # الترخيص
├── .gitignore                 # ملفات Git المستبعدة
│
├── data/                      # مجلد البيانات
│   └── demo_data.csv          # بيانات تجريبية
│
├── www/                       # موارد الويب (CSS, JS, صور)
│   └── (ملفات ثابتة)
│
└── reports/                   # قوالب التقارير
    └── (قوالب RMarkdown)
```

---

## الملفات الرئيسية

### 1. `app.R` - التطبيق الرئيسي

**المحتوى:**
- واجهة المستخدم (UI) بالعربية الكاملة
- منطق الخادم (Server) للتفاعلات
- التبويبات السبعة الرئيسية
- معالجة الأحداث والتفاعلات

**الأقسام الرئيسية:**

```r
# واجهة المستخدم
ui <- fluidPage(
  # CSS للعربية
  # العنوان الرئيسي
  # الشريط الجانبي (سайدبار)
  # التبويبات الرئيسية
)

# الخادم
server <- function(input, output, session) {
  # متغيرات تفاعلية
  # معالجات الأحداث
  # المخرجات
}

# تشغيل التطبيق
shinyApp(ui, server)
```

**التبويبات:**
1. البيانات - عرض ومعاينة
2. الاستقرارية - اختبارات جذر الوحدة
3. مساحة البحث - معلومات البحث
4. مقارنة النماذج - جدول المقارنة
5. النموذج الأفضل - تفاصيل كاملة
6. التشخيصات - اختبارات ورسوم
7. التصدير - تحميل النتائج

---

### 2. `helpers.R` - الدوال المساعدة

**المحتوى:** 13 دالة رئيسية

```r
# 1. استيراد البيانات
data_ingest(file_path, file_type)

# 2. الفهرس الزمني
time_index_resolver(data, date_col, start_date, frequency)

# 3. اختبارات الاستقرارية
unit_root_tests(series, var_name, alpha, max_lag)

# 4. شبكة البحث
build_search_grid(dep_var, indep_vars, max_p, max_q, deterministics)

# 5. تقدير نموذج واحد
fit_ardl_candidate(data, dep_var, indep_vars, p, q_vector, deterministic)

# 6. التشخيصات الشاملة
diagnostics_bundle(model, alpha, bg_order)

# 7. Bounds Test
bounds_test_wrapper(model, case_num, alpha)

# 8. استخراج ECT
ect_extract(model, alpha)

# 9. اختبارات الاستقرار
stability_tests(model, alpha)

# 10. فحص الإشارات الاقتصادية
economic_sign_check(model, expected_signs)

# 11. التقييم
score_model(model, diagnostics, ect_result, ...)

# 12. الترتيب
rank_and_summarize(models_results)

# 13. البيانات التجريبية
generate_synthetic_data(n, seed)
```

**دوال الرسوم:**
```r
plot_time_series(data, var_name)
plot_acf_pacf(series, var_name)
plot_residuals(model)
```

---

### 3. `install.R` - تثبيت الحزم

**الوظيفة:**
- فحص الحزم المثبتة
- تثبيت المفقودة تلقائياً
- التحقق من الحزم الأساسية
- عرض معلومات النظام

**الحزم المطلوبة:**

| الفئة | الحزم |
|-------|-------|
| Shiny | shiny, DT, shinythemes, shinyWidgets |
| البيانات | readr, readxl, writexl, data.table |
| السلاسل الزمنية | zoo, xts, forecast, tseries |
| ARDL | ARDL, dynamac |
| الاختبارات | lmtest, sandwich, urca, car, strucchange, FinTS |
| الرسوم | ggplot2, gridExtra, plotly |
| التقارير | rmarkdown, knitr, officer, flextable |

**الاستخدام:**
```r
source("install.R")
```

---

### 4. `standalone_reestimate.R` - سكربت مستقل

**الوظيفة:**
- إعادة تقدير النموذج الفائز خارج Shiny
- تقرير تفصيلي كامل
- حفظ النتائج والرسوم

**المخرجات:**
```
best_model.rds                # كائن النموذج
best_model_summary.txt        # ملخص نصي
best_model_diagnostics.pdf    # رسوم تشخيصية
```

**خطوات التنفيذ:**
1. تحميل البيانات
2. إنشاء الفهرس الزمني
3. اختبارات الاستقرارية
4. تقدير النموذج
5. التشخيصات الشاملة
6. Bounds Test
7. ECT
8. معاملات الأجل الطويل/القصير
9. اختبارات الاستقرار
10. الرسوم
11. حفظ النتائج

**الاستخدام:**
```r
# عدّل المواصفات في السكربت
BEST_P <- 2
BEST_Q <- c(1, 2)
BEST_DETERMINISTIC <- "const"

# ثم شغّل
source("standalone_reestimate.R")
```

---

## الوثائق

### `README.md` - الدليل الكامل

**الأقسام:**
- نظرة عامة والمبدأ الأساسي
- الميزات الرئيسية
- المتطلبات
- التثبيت والتشغيل
- دليل الاستخدام المفصل
- نظام التقييم
- البيانات التجريبية
- الأسئلة الشائعة
- البنية التقنية
- المراجع العلمية

**الجمهور:** جميع المستخدمين

---

### `QUICKSTART.md` - البدء السريع

**الأقسام:**
- البدء خلال 5 دقائق
- مثال عملي خطوة بخطوة
- نصائح مهمة
- الأخطاء الشائعة
- أسئلة متقدمة

**الجمهور:** مستخدمون جدد، يريدون البدء بسرعة

---

### `TROUBLESHOOTING.md` - حل المشاكل

**الأقسام:**
- مشاكل التثبيت (7 مشاكل)
- مشاكل التشغيل (6 مشاكل)
- مشاكل البيانات (3 مشاكل)
- مشاكل التقدير (13 مشكلة)
- مشاكل الأداء (3 مشاكل)
- مشاكل الرسوم (3 مشاكل)
- مشاكل التصدير (2 مشكلتين)
- مشاكل التفسير (3 مشاكل)

**الجمهور:** مستخدمون يواجهون مشاكل

---

### `PROJECT_STRUCTURE.md` - هذا الملف

**الوظيفة:**
- شرح بنية المشروع
- توثيق الملفات والدوال
- إرشادات التطوير

**الجمهور:** مطورون، مساهمون

---

## مجلد `data/`

### `demo_data.csv` - بيانات تجريبية

**المحتوى:**
- 100 مشاهدة (2000-2099)
- 3 متغيرات: Y, X1, X2
- علاقة تكامل مشترك مصممة

**الاستخدام:**
- اختبار التطبيق
- التعلم
- أمثلة توضيحية

---

## ملفات التكوين

### `DESCRIPTION` - وصف الحزمة

**المحتوى:**
- معلومات الحزمة
- الإصدار والتاريخ
- المؤلفون
- الاعتماديات (Dependencies)
- الترخيص

### `LICENSE` - الترخيص

**النوع:** MIT License

**الحقوق:**
- استخدام حر للأغراض الأكاديمية
- إمكانية التعديل والتوزيع
- بدون ضمانات

**الاقتباس الأكاديمي:**
```
ARDL Shiny App (2025). Version 1.0.0.

Pesaran et al. (2001). Bounds testing approaches...
```

### `.gitignore` - الملفات المستبعدة

**المستبعد:**
- ملفات R المؤقتة (.Rhistory, .RData)
- مخرجات (.pdf, .docx)
- ملفات النظام (.DS_Store)
- ملفات IDE

---

## سير العمل (Workflow)

### للمستخدم العادي:

```
1. تثبيت الحزم (install.R)
   ↓
2. تشغيل التطبيق (app.R)
   ↓
3. رفع البيانات
   ↓
4. تحديد المتغيرات والإعدادات
   ↓
5. تشغيل التحليل
   ↓
6. استعراض النتائج
   ↓
7. تصدير التقارير
```

### للباحث المتقدم:

```
1. تشغيل التطبيق للاستكشاف الأولي
   ↓
2. تحديد النموذج الأفضل
   ↓
3. استخدام standalone_reestimate.R
   ↓
4. تعديل المواصفات يدوياً
   ↓
5. تطبيق اختبارات إضافية
   ↓
6. استخدام النموذج المحفوظ (.rds)
```

### للمطور:

```
1. قراءة PROJECT_STRUCTURE.md
   ↓
2. فهم helpers.R
   ↓
3. تعديل أو إضافة دوال
   ↓
4. تحديث app.R إذا لزم
   ↓
5. اختبار التغييرات
   ↓
6. توثيق التعديلات
   ↓
7. تحديث DESCRIPTION
```

---

## معمارية الكود

### طبقة البيانات
```
data_ingest() → time_index_resolver()
                     ↓
                data_ts (zoo/ts object)
```

### طبقة الاختبارات
```
unit_root_tests() → للمتغيرات الفردية
                     ↓
              stationarity_results
```

### طبقة النمذجة
```
build_search_grid() → fit_ardl_candidate() → diagnostics_bundle()
                                                    ↓
                                             all_models
```

### طبقة التقييم
```
bounds_test_wrapper() → ect_extract() → stability_tests() → score_model()
                                                                  ↓
                                                         ranked_models
```

### طبقة العرض
```
rank_and_summarize() → Shiny outputs
                            ↓
                    Tables, Plots, Reports
```

---

## نقاط التوسع المستقبلية

### ميزات مخططة:

1. **NARDL (Non-linear ARDL)**
   - ملف: `nardl_functions.R`
   - إضافة تبويب جديد

2. **التنبؤ (Forecasting)**
   - ملف: `forecast_functions.R`
   - تبويب "التنبؤ"

3. **تقارير LaTeX**
   - استخدام knitr + LaTeX
   - قوالب أكاديمية

4. **قاعدة بيانات للنتائج**
   - SQLite لحفظ التحليلات
   - مقارنة عبر الزمن

5. **واجهة متعددة اللغات**
   - ملفات i18n
   - دعم EN, FR, ES

### التحسينات التقنية:

1. **التخزين المؤقت (Caching)**
   ```r
   library(memoise)
   fit_ardl_cached <- memoise(fit_ardl_candidate)
   ```

2. **التوازي المتقدم**
   ```r
   library(future)
   library(promises)
   ```

3. **اختبارات الوحدة (Unit Tests)**
   ```r
   library(testthat)
   test_that("unit_root_tests works", { ... })
   ```

4. **CI/CD**
   - GitHub Actions
   - اختبار تلقائي عند كل commit

---

## إرشادات المساهمة

### إضافة دالة جديدة:

1. أضفها في `helpers.R` مع:
   - تعليق توضيحي
   - معالجة الأخطاء (tryCatch)
   - قيم افتراضية منطقية

2. وثّقها في `PROJECT_STRUCTURE.md`

3. أضف مثال استخدام في `QUICKSTART.md`

4. إذا كانت معقدة، أضف troubleshooting في `TROUBLESHOOTING.md`

### تعديل واجهة المستخدم:

1. عدّل قسم UI في `app.R`

2. أضف معالج في قسم Server

3. اختبر على متصفحات مختلفة

4. تأكد من دعم العربية

---

## الاختبار

### اختبار يدوي سريع:

```r
# 1. التثبيت
source("install.R")

# 2. البيانات التجريبية
shiny::runApp("app.R")
# ثم:
# - فعّل "استخدام بيانات تجريبية"
# - تشغيل التحليل
# - تحقق من النتائج

# 3. السكربت المستقل
source("standalone_reestimate.R")
# - تحقق من المخرجات
```

### اختبار متقدم:

```r
# اختبار كل دالة على حدة
source("helpers.R")

# بيانات اختبار
test_data <- generate_synthetic_data(50)

# اختبار استقرارية
result <- unit_root_tests(test_data$Y, "Y", 0.05)
print(result)

# اختبار تقدير
# ... وهكذا
```

---

## الصيانة

### تحديث الحزم:

```r
# فحص التحديثات
old.packages()

# تحديث الكل
update.packages(ask = FALSE)

# إعادة تشغيل install.R للتأكد
source("install.R")
```

### تحديث الإصدار:

1. عدّل `DESCRIPTION`:
   ```
   Version: 1.1.0
   Date: 2025-XX-XX
   ```

2. أضف سجل في `README.md` تحت "ملاحظات الإصدار"

3. commit مع رسالة واضحة:
   ```
   git commit -m "v1.1.0: إضافة ميزة X وإصلاح Y"
   ```

---

## معلومات الإصدار

- **الإصدار الحالي:** 1.0.0
- **تاريخ الإنشاء:** 2025-10-22
- **آخر تحديث:** 2025-10-22
- **الحالة:** مستقر (Stable)

---

## الاتصال والدعم

للأسئلة أو الاقتراحات:
- افتح Issue على GitHub
- راجع TROUBLESHOOTING.md أولاً
- تحقق من الوثائق

---

**نهاية ملف بنية المشروع**
