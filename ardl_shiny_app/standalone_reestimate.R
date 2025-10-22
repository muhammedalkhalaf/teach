# ============================================================================
# سكربت مستقل لإعادة تقدير النموذج الفائز
# Standalone Script to Re-estimate the Best Model
# ============================================================================

# تحميل الدوال المساعدة
source("helpers.R")

cat("=" , rep("=", 70), "\n")
cat("سكربت إعادة تقدير النموذج الفائز - ARDL Model\n")
cat("Standalone Re-estimation Script\n")
cat(rep("=", 70), "\n\n")

# ============================================================================
# 1. الإعدادات الأساسية
# ============================================================================

# تحميل البيانات
cat("1. تحميل البيانات...\n")

# استخدم بياناتك الخاصة أو البيانات التجريبية
USE_DEMO_DATA <- TRUE

if (USE_DEMO_DATA) {
  cat("   استخدام البيانات التجريبية...\n")
  data_raw <- generate_synthetic_data(n = 100, seed = 123)
} else {
  cat("   تحميل بيانات من ملف...\n")
  # عدّل المسار إلى ملف بياناتك
  data_raw <- read.csv("data/your_data.csv")
}

cat("   ✓ تم تحميل", nrow(data_raw), "مشاهدة\n\n")

# ============================================================================
# 2. إنشاء الفهرس الزمني
# ============================================================================

cat("2. إنشاء الفهرس الزمني...\n")

data_ts <- time_index_resolver(
  data = data_raw,
  date_col = NULL,  # أو اسم عمود التاريخ
  start_date = c(2000, 1),
  frequency = "annual"
)

cat("   ✓ تم إنشاء الفهرس الزمني\n\n")

# ============================================================================
# 3. تحديد المتغيرات
# ============================================================================

cat("3. تحديد المتغيرات...\n")

# عدّل هذه حسب بياناتك
DEP_VAR <- "Y"
INDEP_VARS <- c("X1", "X2")

cat("   المتغير التابع:", DEP_VAR, "\n")
cat("   المتغيرات المستقلة:", paste(INDEP_VARS, collapse = ", "), "\n\n")

# ============================================================================
# 4. اختبارات الاستقرارية
# ============================================================================

cat("4. اختبارات الاستقرارية...\n\n")

all_vars <- c(DEP_VAR, INDEP_VARS)
stationarity_results <- list()

for (var in all_vars) {
  cat("   اختبار", var, "...\n")

  result <- unit_root_tests(
    series = as.numeric(data_ts[, var]),
    var_name = var,
    alpha = 0.05
  )

  stationarity_results[[var]] <- result

  # عرض نتيجة مختصرة
  adf_decision <- if (!is.null(result$adf$drift)) result$adf$drift$decision else "N/A"
  pp_decision <- if (!is.null(result$pp)) result$pp$decision else "N/A"

  cat("      ADF (drift):", adf_decision, "\n")
  cat("      PP:", pp_decision, "\n\n")
}

cat("   ✓ اكتملت اختبارات الاستقرارية\n\n")

# ============================================================================
# 5. مواصفات النموذج الأفضل
# ============================================================================

cat("5. مواصفات النموذج الأفضل...\n")

# عدّل هذه حسب النموذج الذي اخترته من التطبيق
BEST_P <- 2
BEST_Q <- c(1, 2)  # لكل متغير مستقل
BEST_DETERMINISTIC <- "const"

cat("   ARDL(", BEST_P, ", ", paste(BEST_Q, collapse = ", "), ")\n", sep = "")
cat("   المكونات الحتمية:", BEST_DETERMINISTIC, "\n\n")

# ============================================================================
# 6. تقدير النموذج
# ============================================================================

cat("6. تقدير النموذج...\n")

best_model <- fit_ardl_candidate(
  data = as.data.frame(data_ts),
  dep_var = DEP_VAR,
  indep_vars = INDEP_VARS,
  p = BEST_P,
  q_vector = BEST_Q,
  deterministic = BEST_DETERMINISTIC
)

if (is.null(best_model)) {
  stop("✗ فشل تقدير النموذج")
}

cat("   ✓ تم تقدير النموذج بنجاح\n\n")

# ============================================================================
# 7. ملخص النموذج
# ============================================================================

cat("7. ملخص النموذج:\n")
cat(rep("-", 70), "\n")
print(summary(best_model))
cat(rep("-", 70), "\n\n")

# ============================================================================
# 8. التشخيصات
# ============================================================================

cat("8. التشخيصات الشاملة...\n\n")

diagnostics <- diagnostics_bundle(
  model = best_model,
  alpha = 0.05,
  bg_order = 2
)

cat("   اختبارات التشخيص:\n")
cat("   -------------------\n")
cat("   Breusch-Godfrey:\n")
cat("      الإحصاء:", round(diagnostics$bg$statistic, 4), "\n")
cat("      القيمة الاحتمالية:", round(diagnostics$bg$p_value, 4), "\n")
cat("      النتيجة:", ifelse(diagnostics$bg$passed, "✓ ناجح", "✗ فاشل"), "\n\n")

cat("   Breusch-Pagan:\n")
cat("      الإحصاء:", round(diagnostics$bp$statistic, 4), "\n")
cat("      القيمة الاحتمالية:", round(diagnostics$bp$p_value, 4), "\n")
cat("      النتيجة:", ifelse(diagnostics$bp$passed, "✓ ناجح", "✗ فاشل"), "\n\n")

cat("   Jarque-Bera:\n")
cat("      الإحصاء:", round(diagnostics$jb$statistic, 4), "\n")
cat("      القيمة الاحتمالية:", round(diagnostics$jb$p_value, 4), "\n")
cat("      النتيجة:", ifelse(diagnostics$jb$passed, "✓ ناجح", "✗ فاشل"), "\n\n")

cat("   معايير المعلومات:\n")
cat("      AIC:", round(diagnostics$aic, 4), "\n")
cat("      BIC:", round(diagnostics$bic, 4), "\n\n")

# ============================================================================
# 9. Bounds Test
# ============================================================================

cat("9. Bounds Test للتكامل المشترك...\n\n")

case_num <- switch(BEST_DETERMINISTIC,
                   "none" = 1, "const" = 3, "trend" = 2, "both" = 5, 3)

bounds <- bounds_test_wrapper(
  model = best_model,
  case_num = case_num,
  alpha = 0.05
)

if (!is.null(bounds) && !bounds$failed) {
  cat("   F-Statistic:", round(bounds$f_statistic, 4), "\n")
  cat("   Lower Bound (5%):", round(bounds$lower_bound, 4), "\n")
  cat("   Upper Bound (5%):", round(bounds$upper_bound, 4), "\n")
  cat("   القرار:", bounds$decision, "\n\n")
} else {
  cat("   ✗ فشل Bounds Test\n\n")
}

# ============================================================================
# 10. معامل تصحيح الخطأ (ECT)
# ============================================================================

cat("10. معامل تصحيح الخطأ (ECT)...\n\n")

ect <- ect_extract(
  model = best_model,
  alpha = 0.05
)

if (ect$passed) {
  cat("   المعامل:", round(ect$coefficient, 4), "\n")
  cat("   الانحراف المعياري:", round(ect$std_error, 4), "\n")
  cat("   إحصاء t:", round(ect$t_statistic, 4), "\n")
  cat("   القيمة الاحتمالية:", round(ect$p_value, 4), "\n")
  cat("   النتيجة:", ifelse(ect$passed, "✓ سالب ومعنوي", "✗ فاشل"), "\n\n")
} else {
  cat("   ✗ فشل استخراج ECT\n\n")
}

# ============================================================================
# 11. معاملات الأجل الطويل
# ============================================================================

cat("11. معاملات الأجل الطويل (Long-Run Coefficients)...\n\n")

lr_coefs <- tryCatch({
  multipliers(best_model, type = "lr")
}, error = function(e) NULL)

if (!is.null(lr_coefs)) {
  cat("   المتغير         |   المعامل  | الانحراف | إحصاء t |  p-value\n")
  cat("   ", rep("-", 65), "\n")

  for (i in 1:nrow(lr_coefs)) {
    var_name <- rownames(lr_coefs)[i]
    estimate <- lr_coefs$estimate[i]
    std_err <- lr_coefs$std.error[i]
    t_stat <- lr_coefs$statistic[i]
    p_val <- lr_coefs$p.value[i]

    cat(sprintf("   %-15s | %10.4f | %9.4f | %8.4f | %8.4f %s\n",
                var_name, estimate, std_err, t_stat, p_val,
                ifelse(p_val < 0.05, "*", "")))
  }
  cat("\n")
} else {
  cat("   ✗ فشل استخراج معاملات الأجل الطويل\n\n")
}

# ============================================================================
# 12. معاملات الأجل القصير
# ============================================================================

cat("12. معاملات الأجل القصير (Short-Run Coefficients)...\n\n")

ecm_model <- recm(best_model, case = best_model$case)
sr_coefs <- summary(ecm_model)$coefficients

cat("   المتغير         |   المعامل  | الانحراف | إحصاء t |  p-value\n")
cat("   ", rep("-", 65), "\n")

for (i in 1:nrow(sr_coefs)) {
  var_name <- rownames(sr_coefs)[i]
  estimate <- sr_coefs[i, "Estimate"]
  std_err <- sr_coefs[i, "Std. Error"]
  t_stat <- sr_coefs[i, "t value"]
  p_val <- sr_coefs[i, "Pr(>|t|)"]

  cat(sprintf("   %-15s | %10.4f | %9.4f | %8.4f | %8.4f %s\n",
              var_name, estimate, std_err, t_stat, p_val,
              ifelse(p_val < 0.05, "*", "")))
}
cat("\n")

# ============================================================================
# 13. اختبارات الاستقرار
# ============================================================================

cat("13. اختبارات الاستقرار (CUSUM & CUSUMSQ)...\n\n")

stability <- stability_tests(
  model = best_model,
  alpha = 0.05
)

if (!is.null(stability) && !is.null(stability$cusum)) {
  cat("   CUSUM:", ifelse(stability$cusum$passed, "✓ ناجح", "✗ فاشل"), "\n")
  cat("      القيمة الاحتمالية:", round(stability$cusum$p_value, 4), "\n\n")

  cat("   CUSUMSQ:", ifelse(stability$cusumsq$passed, "✓ ناجح", "✗ فاشل"), "\n")
  cat("      القيمة الاحتمالية:", round(stability$cusumsq$p_value, 4), "\n\n")

  cat("   النتيجة الكلية:", ifelse(stability$both_passed, "✓ مستقر", "✗ غير مستقر"), "\n\n")
} else {
  cat("   ✗ فشل اختبار الاستقرار\n\n")
}

# ============================================================================
# 14. الرسوم البيانية
# ============================================================================

cat("14. توليد الرسوم البيانية...\n\n")

# فتح جهاز رسم PDF
pdf("best_model_diagnostics.pdf", width = 10, height = 8)

# البواقي
cat("   رسم البواقي...\n")
plot_residuals(best_model)

# CUSUM
cat("   رسم CUSUM...\n")
ols_model <- lm(best_model$formula, data = best_model$model)

par(mfrow = c(1, 2))
cusum <- efp(ols_model, type = "Rec-CUSUM")
plot(cusum, main = "CUSUM Test")

cusumsq <- efp(ols_model, type = "OLS-CUSUM")
plot(cusumsq, main = "CUSUMSQ Test")

# القيم الفعلية مقابل الملائمة
cat("   رسم القيم الفعلية مقابل الملائمة...\n")
par(mfrow = c(1, 1))
actual <- best_model$model[, 1]
fitted_vals <- fitted(best_model)

plot(actual, type = "l", col = "black", lwd = 2,
     main = "القيم الفعلية مقابل القيم الملائمة",
     xlab = "المشاهدة", ylab = "القيمة", ylim = range(c(actual, fitted_vals)))
lines(fitted_vals, col = "red", lwd = 2, lty = 2)
legend("topleft", legend = c("فعلي", "ملائم"),
       col = c("black", "red"), lty = c(1, 2), lwd = 2)
grid()

dev.off()

cat("   ✓ تم حفظ الرسوم في: best_model_diagnostics.pdf\n\n")

# ============================================================================
# 15. حفظ النتائج
# ============================================================================

cat("15. حفظ النتائج...\n\n")

# حفظ كائن النموذج
saveRDS(best_model, file = "best_model.rds")
cat("   ✓ تم حفظ النموذج في: best_model.rds\n")

# حفظ ملخص كنص
sink("best_model_summary.txt")
cat("=" , rep("=", 70), "\n")
cat("ملخص النموذج الأفضل - ARDL Model\n")
cat(rep("=", 70), "\n\n")

cat("المواصفات:\n")
cat("  ARDL(", BEST_P, ", ", paste(BEST_Q, collapse = ", "), ")\n", sep = "")
cat("  المكونات الحتمية:", BEST_DETERMINISTIC, "\n\n")

cat("الملخص الكامل:\n")
print(summary(best_model))

cat("\n\n")
cat("معاملات الأجل الطويل:\n")
print(lr_coefs)

cat("\n\n")
cat("معاملات الأجل القصير:\n")
print(sr_coefs)

sink()

cat("   ✓ تم حفظ الملخص في: best_model_summary.txt\n\n")

# ============================================================================
# 16. النهاية
# ============================================================================

cat(rep("=", 70), "\n")
cat("✓✓✓ اكتمل التحليل بنجاح! ✓✓✓\n")
cat(rep("=", 70), "\n\n")

cat("الملفات الناتجة:\n")
cat("  1. best_model.rds - كائن النموذج\n")
cat("  2. best_model_summary.txt - ملخص نصي\n")
cat("  3. best_model_diagnostics.pdf - الرسوم البيانية\n\n")

cat("لإعادة تحميل النموذج لاحقاً:\n")
cat("  model <- readRDS('best_model.rds')\n\n")

# معلومات الجلسة
cat("معلومات الجلسة:\n")
print(sessionInfo())
