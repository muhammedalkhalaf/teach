# ============================================================================
# ARDL Model Selection - Helper Functions
# نظام الاختيار الآلي لنماذج ARDL - الدوال المساعدة
# ============================================================================

# تحميل المكتبات المطلوبة
suppressPackageStartupMessages({
  library(shiny)
  library(readr)
  library(readxl)
  library(data.table)
  library(zoo)
  library(xts)
  library(ARDL)
  library(lmtest)
  library(sandwich)
  library(tseries)
  library(urca)
  library(car)
  library(strucchange)
  library(ggplot2)
  library(parallel)
  library(DT)
  library(rmarkdown)
  library(officer)
  library(flextable)
})

# ============================================================================
# 1. دوال استيراد البيانات وفهرسة الزمن
# ============================================================================

#' استيراد البيانات من CSV أو Excel
data_ingest <- function(file_path, file_type = "csv") {
  tryCatch({
    if (file_type == "csv") {
      data <- read_csv(file_path, show_col_types = FALSE)
    } else if (file_type == "xlsx") {
      data <- read_excel(file_path)
    } else {
      stop("نوع الملف غير مدعوم")
    }
    return(as.data.frame(data))
  }, error = function(e) {
    stop(paste("خطأ في قراءة الملف:", e$message))
  })
}

#' إنشاء فهرس زمني للبيانات
time_index_resolver <- function(data, date_col = NULL, start_date = NULL,
                                frequency = "annual") {
  tryCatch({
    n <- nrow(data)

    if (!is.null(date_col) && date_col %in% names(data)) {
      # استخدام عمود التاريخ الموجود
      dates <- as.Date(data[[date_col]])
      data_ts <- zoo(data[, -which(names(data) == date_col), drop = FALSE],
                     order.by = dates)
    } else {
      # إنشاء فهرس زمني جديد
      freq_map <- list(
        "annual" = 1,
        "quarterly" = 4,
        "monthly" = 12
      )

      freq_num <- freq_map[[frequency]]
      if (is.null(start_date)) start_date <- c(2000, 1)

      data_ts <- ts(data, start = start_date, frequency = freq_num)
    }

    return(data_ts)
  }, error = function(e) {
    stop(paste("خطأ في إنشاء الفهرس الزمني:", e$message))
  })
}

# ============================================================================
# 2. اختبارات جذر الوحدة (Stationarity Tests)
# ============================================================================

#' تنفيذ اختبارات الاستقرارية الشاملة
unit_root_tests <- function(series, var_name, alpha = 0.05, max_lag = NULL) {

  if (is.null(max_lag)) {
    max_lag <- floor(12 * (length(series)/100)^0.25)
  }

  results <- list()

  # ADF Test - ثلاث حالات
  adf_none <- tryCatch({
    ur.df(series, type = "none", lags = max_lag, selectlags = "AIC")
  }, error = function(e) NULL)

  adf_drift <- tryCatch({
    ur.df(series, type = "drift", lags = max_lag, selectlags = "AIC")
  }, error = function(e) NULL)

  adf_trend <- tryCatch({
    ur.df(series, type = "trend", lags = max_lag, selectlags = "AIC")
  }, error = function(e) NULL)

  # PP Test
  pp_result <- tryCatch({
    PP.test(series)
  }, error = function(e) NULL)

  # KPSS Test - حالتين
  kpss_level <- tryCatch({
    ur.kpss(series, type = "mu", lags = "short")
  }, error = function(e) NULL)

  kpss_trend <- tryCatch({
    ur.kpss(series, type = "tau", lags = "short")
  }, error = function(e) NULL)

  # تنسيق النتائج
  results$variable <- var_name
  results$adf <- list(
    none = if (!is.null(adf_none)) {
      list(
        statistic = adf_none@teststat[1],
        critical_values = adf_none@cval[1, ],
        decision = ifelse(adf_none@teststat[1] < adf_none@cval[1, 2], "مستقر", "غير مستقر")
      )
    } else NULL,
    drift = if (!is.null(adf_drift)) {
      list(
        statistic = adf_drift@teststat[1],
        critical_values = adf_drift@cval[1, ],
        decision = ifelse(adf_drift@teststat[1] < adf_drift@cval[1, 2], "مستقر", "غير مستقر")
      )
    } else NULL,
    trend = if (!is.null(adf_trend)) {
      list(
        statistic = adf_trend@teststat[1],
        critical_values = adf_trend@cval[1, ],
        decision = ifelse(adf_trend@teststat[1] < adf_trend@cval[1, 2], "مستقر", "غير مستقر")
      )
    } else NULL
  )

  results$pp <- if (!is.null(pp_result)) {
    list(
      statistic = pp_result$statistic,
      p_value = pp_result$p.value,
      decision = ifelse(pp_result$p.value < alpha, "مستقر", "غير مستقر")
    )
  } else NULL

  results$kpss <- list(
    level = if (!is.null(kpss_level)) {
      list(
        statistic = kpss_level@teststat[1],
        critical_values = kpss_level@cval[1, ],
        decision = ifelse(kpss_level@teststat[1] < kpss_level@cval[1, 2], "مستقر", "غير مستقر")
      )
    } else NULL,
    trend = if (!is.null(kpss_trend)) {
      list(
        statistic = kpss_trend@teststat[1],
        critical_values = kpss_trend@cval[1, ],
        decision = ifelse(kpss_trend@teststat[1] < kpss_trend@cval[1, 2], "مستقر", "غير مستقر")
      )
    } else NULL
  )

  return(results)
}

# ============================================================================
# 3. بناء شبكة البحث للنماذج المرشحة
# ============================================================================

#' توليد كل مجموعات النماذج المرشحة
build_search_grid <- function(dep_var, indep_vars, max_p = 4, max_q = 4,
                               deterministics = c("const", "trend", "both", "none")) {

  # توليد مجموعات تأخيرات المتغيرات
  p_range <- 0:max_p

  # لكل متغير مستقل، نطاق من 0 إلى max_q
  q_combinations <- expand.grid(lapply(indep_vars, function(x) 0:max_q))
  names(q_combinations) <- indep_vars

  # دمج مع p والمكونات الحتمية
  grid <- expand.grid(
    p = p_range,
    deterministic = deterministics,
    stringsAsFactors = FALSE
  )

  # إضافة q's لكل متغير مستقل
  grid <- cbind(grid, q_combinations[rep(1:nrow(q_combinations),
                                         each = nrow(grid)), ])

  # إضافة معرف فريد
  grid$model_id <- seq_len(nrow(grid))

  return(grid)
}

# ============================================================================
# 4. تقدير نموذج ARDL مرشح واحد
# ============================================================================

#' تقدير نموذج ARDL واحد
fit_ardl_candidate <- function(data, dep_var, indep_vars, p, q_vector,
                               deterministic = "const") {

  tryCatch({
    # بناء الصيغة
    formula_str <- paste0(dep_var, " ~ ")

    # إضافة المتغيرات المستقلة مع تأخيراتها
    for (i in seq_along(indep_vars)) {
      if (i > 1) formula_str <- paste0(formula_str, " + ")
      formula_str <- paste0(formula_str, indep_vars[i])
    }

    # تحديد ترتيب ARDL
    order_vec <- c(p, q_vector)
    names(order_vec) <- c(dep_var, indep_vars)

    # تقدير النموذج باستخدام حزمة ARDL
    case_num <- switch(deterministic,
                       "none" = 1,
                       "const" = 3,
                       "trend" = 2,
                       "both" = 5,
                       3)

    model <- ARDL::ardl(
      formula = as.formula(formula_str),
      data = as.data.frame(data),
      order = order_vec,
      case = case_num
    )

    return(model)

  }, error = function(e) {
    return(NULL)
  })
}

# ============================================================================
# 5. حزمة التشخيصات الشاملة
# ============================================================================

#' تنفيذ جميع الاختبارات التشخيصية
diagnostics_bundle <- function(model, alpha = 0.05, bg_order = 2) {

  if (is.null(model)) {
    return(list(failed = TRUE, reason = "فشل التقدير"))
  }

  results <- list()
  results$failed <- FALSE

  tryCatch({
    # استخراج البواقي
    residuals_vec <- residuals(model)

    # 1. Breusch-Godfrey Test للارتباط الذاتي
    bg_test <- bgtest(model, order = bg_order)
    results$bg <- list(
      statistic = bg_test$statistic,
      p_value = bg_test$p.value,
      passed = bg_test$p.value > alpha
    )

    # 2. Breusch-Pagan Test لعدم تجانس التباين
    bp_test <- bptest(model)
    results$bp <- list(
      statistic = bp_test$statistic,
      p_value = bp_test$p.value,
      passed = bp_test$p.value > alpha
    )

    # 3. ARCH Test
    arch_test <- tryCatch({
      ArchTest(residuals_vec, lags = 2)
    }, error = function(e) NULL)

    results$arch <- if (!is.null(arch_test)) {
      list(
        statistic = arch_test$statistic,
        p_value = arch_test$p.value,
        passed = arch_test$p.value > alpha
      )
    } else {
      list(passed = TRUE)
    }

    # 4. Jarque-Bera Test للتوزيع الطبيعي
    jb_test <- jarque.bera.test(residuals_vec)
    results$jb <- list(
      statistic = jb_test$statistic,
      p_value = jb_test$p.value,
      passed = jb_test$p.value > alpha
    )

    # 5. CUSUM و CUSUMSQ للاستقرار
    ols_model <- lm(model$formula, data = model$model)

    cusum_test <- tryCatch({
      efp(ols_model$model[, 1] ~ ., data = ols_model$model[, -1, drop = FALSE],
          type = "Rec-CUSUM")
    }, error = function(e) NULL)

    cusumsq_test <- tryCatch({
      efp(ols_model$model[, 1] ~ ., data = ols_model$model[, -1, drop = FALSE],
          type = "OLS-CUSUM")
    }, error = function(e) NULL)

    results$cusum <- if (!is.null(cusum_test)) {
      test_result <- sctest(cusum_test)
      list(
        statistic = test_result$statistic,
        p_value = test_result$p.value,
        passed = test_result$p.value > alpha
      )
    } else {
      list(passed = TRUE)
    }

    results$cusumsq <- if (!is.null(cusumsq_test)) {
      test_result <- sctest(cusumsq_test)
      list(
        statistic = test_result$statistic,
        p_value = test_result$p.value,
        passed = test_result$p.value > alpha
      )
    } else {
      list(passed = TRUE)
    }

    # 6. معايير المعلومات
    results$aic <- AIC(model)
    results$bic <- BIC(model)

    return(results)

  }, error = function(e) {
    return(list(failed = TRUE, reason = paste("خطأ في التشخيصات:", e$message)))
  })
}

# ============================================================================
# 6. Bounds Test للتكامل المشترك
# ============================================================================

#' تنفيذ Bounds Test
bounds_test_wrapper <- function(model, case_num = 3, alpha = 0.05) {

  tryCatch({
    # استخدام bounds_f_test من حزمة ARDL
    bounds_result <- bounds_f_test(model, case = case_num, alpha = alpha)

    result <- list(
      f_statistic = bounds_result$tab[1, "statistic"],
      lower_bound = bounds_result$tab[1, paste0("Lower, ", alpha)],
      upper_bound = bounds_result$tab[1, paste0("Upper, ", alpha)],
      decision = if (bounds_result$tab[1, "statistic"] > bounds_result$tab[1, paste0("Upper, ", alpha)]) {
        "يوجد تكامل مشترك"
      } else if (bounds_result$tab[1, "statistic"] < bounds_result$tab[1, paste0("Lower, ", alpha)]) {
        "لا يوجد تكامل مشترك"
      } else {
        "غير حاسم"
      },
      passed = bounds_result$tab[1, "statistic"] > bounds_result$tab[1, paste0("Upper, ", alpha)]
    )

    return(result)

  }, error = function(e) {
    return(list(failed = TRUE, reason = paste("فشل Bounds Test:", e$message)))
  })
}

# ============================================================================
# 7. استخراج معامل تصحيح الخطأ (ECT)
# ============================================================================

#' استخراج معامل ECT وإحصاءاته
ect_extract <- function(model, alpha = 0.05) {

  tryCatch({
    # الحصول على تمثيل تصحيح الخطأ
    ecm_model <- recm(model, case = model$case)

    # استخراج معامل ECT
    coefs <- summary(ecm_model)$coefficients
    ect_row <- grep("^ect$|^L1\\.", rownames(coefs))[1]

    if (length(ect_row) > 0 && !is.na(ect_row)) {
      ect_coef <- coefs[ect_row, "Estimate"]
      ect_se <- coefs[ect_row, "Std. Error"]
      ect_t <- coefs[ect_row, "t value"]
      ect_p <- coefs[ect_row, "Pr(>|t|)"]

      result <- list(
        coefficient = ect_coef,
        std_error = ect_se,
        t_statistic = ect_t,
        p_value = ect_p,
        is_negative = ect_coef < 0,
        is_significant = ect_p < alpha,
        passed = (ect_coef < 0) && (ect_p < alpha)
      )
    } else {
      result <- list(passed = FALSE, reason = "لم يتم العثور على ECT")
    }

    return(result)

  }, error = function(e) {
    return(list(passed = FALSE, reason = paste("فشل استخراج ECT:", e$message)))
  })
}

# ============================================================================
# 8. اختبارات الاستقرار (CUSUM)
# ============================================================================

#' اختبارات CUSUM و CUSUMSQ
stability_tests <- function(model, alpha = 0.05) {

  tryCatch({
    # تحويل إلى نموذج lm للاختبارات
    ols_model <- lm(model$formula, data = model$model)

    # CUSUM
    cusum <- efp(ols_model, type = "Rec-CUSUM")
    cusum_test <- sctest(cusum)

    # CUSUMSQ
    cusumsq <- efp(ols_model, type = "OLS-CUSUM")
    cusumsq_test <- sctest(cusumsq)

    result <- list(
      cusum = list(
        passed = cusum_test$p.value > alpha,
        p_value = cusum_test$p.value
      ),
      cusumsq = list(
        passed = cusumsq_test$p.value > alpha,
        p_value = cusumsq_test$p.value
      ),
      both_passed = (cusum_test$p.value > alpha) && (cusumsq_test$p.value > alpha)
    )

    return(result)

  }, error = function(e) {
    return(list(both_passed = FALSE, reason = paste("فشل اختبار الاستقرار:", e$message)))
  })
}

# ============================================================================
# 9. فحص المنطق الاقتصادي للإشارات
# ============================================================================

#' التحقق من إشارات المعاملات
economic_sign_check <- function(model, expected_signs) {

  tryCatch({
    # استخراج معاملات الأجل الطويل
    lr_model <- multipliers(model, type = "lr")
    coefs <- lr_model$estimate

    matches <- 0
    conflicts <- 0

    for (var_name in names(expected_signs)) {
      if (var_name %in% names(coefs)) {
        actual_sign <- sign(coefs[var_name])
        expected_sign <- expected_signs[var_name]

        if (expected_sign == 0) {
          next  # لا توقع
        } else if (actual_sign == expected_sign) {
          matches <- matches + 1
        } else {
          conflicts <- conflicts + 1
        }
      }
    }

    return(list(matches = matches, conflicts = conflicts))

  }, error = function(e) {
    return(list(matches = 0, conflicts = 0))
  })
}

# ============================================================================
# 10. نظام التقييم والنقاط
# ============================================================================

#' احتساب النقاط الكلية للنموذج
score_model <- function(model, diagnostics, ect_result, stability_result,
                       bounds_result, sign_check, aic_rank, alpha = 0.05) {

  score <- 0
  flags <- list()

  # المرشّحات الصارمة - فشل أي منها = 0

  # 1. الارتباط الذاتي (Breusch-Godfrey)
  if (!diagnostics$bg$passed) {
    flags$bg_failed <- TRUE
    return(list(score = 0, flags = flags, reason = "فشل BG - ارتباط ذاتي"))
  } else {
    flags$bg_passed <- TRUE
  }

  # 2. ECT سالب ومعنوي
  if (!ect_result$passed) {
    flags$ect_failed <- TRUE
    return(list(score = 0, flags = flags, reason = "ECT غير صحيح"))
  } else {
    flags$ect_passed <- TRUE
  }

  # الآن نضيف النقاط للنماذج التي اجتازت المرشّحات

  # 3. عدم تجانس التباين
  if (diagnostics$bp$passed && diagnostics$arch$passed) {
    score <- score + 2
    flags$heteroskedasticity_passed <- TRUE
  } else {
    score <- score - 1
    flags$heteroskedasticity_failed <- TRUE
  }

  # 4. الاستقرار
  if (stability_result$both_passed) {
    score <- score + 2
    flags$stability_passed <- TRUE
  } else if (stability_result$cusum$passed || stability_result$cusumsq$passed) {
    score <- score - 1
    flags$stability_partial <- TRUE
  } else {
    score <- score - 3
    flags$stability_failed <- TRUE
  }

  # 5. معنوية معاملات الأجل الطويل
  lr_coefs <- tryCatch({
    multipliers(model, type = "lr")
  }, error = function(e) NULL)

  if (!is.null(lr_coefs)) {
    sig_count <- sum(lr_coefs$p.value < alpha, na.rm = TRUE)
    score <- score + sig_count
    flags$significant_lr_vars <- sig_count
  }

  # 6. المنطق الاقتصادي للإشارات
  score <- score + sign_check$matches - sign_check$conflicts
  flags$sign_matches <- sign_check$matches
  flags$sign_conflicts <- sign_check$conflicts

  # 7. Bounds Test
  if (!is.null(bounds_result) && bounds_result$passed) {
    score <- score + 1
    flags$bounds_passed <- TRUE
  }

  # 8. ترتيب AIC (مكافأة بسيطة لأفضل 10%)
  if (aic_rank <= 0.1) {
    score <- score + 1
    flags$top_aic <- TRUE
  }

  return(list(score = score, flags = flags))
}

# ============================================================================
# 11. ترتيب النماذج وتلخيص النتائج
# ============================================================================

#' ترتيب النماذج حسب النقاط
rank_and_summarize <- function(models_results) {

  # تصفية النماذج الفاشلة
  valid_models <- Filter(function(x) !is.null(x$score_result) &&
                           x$score_result$score > 0, models_results)

  if (length(valid_models) == 0) {
    return(list(
      top_models = NULL,
      message = "لم يجتاز أي نموذج المرشّحات الصارمة"
    ))
  }

  # ترتيب حسب النقاط (تنازلي) ثم AIC (تصاعدي)
  scores <- sapply(valid_models, function(x) x$score_result$score)
  aics <- sapply(valid_models, function(x) x$diagnostics$aic)

  order_idx <- order(-scores, aics)
  ranked_models <- valid_models[order_idx]

  return(list(
    top_models = ranked_models,
    count = length(ranked_models)
  ))
}

# ============================================================================
# 12. توليد البيانات التجريبية
# ============================================================================

#' توليد بيانات تركيبية لاختبار التطبيق
generate_synthetic_data <- function(n = 100, seed = 123) {
  set.seed(seed)

  # توليد متغيرات I(1)
  e1 <- rnorm(n)
  e2 <- rnorm(n)
  e3 <- rnorm(n)

  x1 <- cumsum(e1)
  x2 <- cumsum(e2)

  # علاقة تكامل مشترك
  y <- 0.5 * x1 + 0.3 * x2 + cumsum(e3 * 0.5)

  data <- data.frame(
    year = 2000:(2000 + n - 1),
    Y = y,
    X1 = x1,
    X2 = x2
  )

  return(data)
}

# ============================================================================
# 13. دوال الرسوم البيانية
# ============================================================================

#' رسم السلاسل الزمنية
plot_time_series <- function(data, var_name) {
  df <- data.frame(
    time = time(data),
    value = as.numeric(data[, var_name])
  )

  ggplot(df, aes(x = time, y = value)) +
    geom_line(color = "steelblue", size = 1) +
    labs(title = paste("السلسلة الزمنية:", var_name),
         x = "الزمن", y = "القيمة") +
    theme_minimal(base_size = 14) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      axis.text = element_text(size = 12)
    )
}

#' رسم ACF و PACF
plot_acf_pacf <- function(series, var_name) {
  par(mfrow = c(1, 2))
  acf(series, main = paste("ACF -", var_name))
  pacf(series, main = paste("PACF -", var_name))
  par(mfrow = c(1, 1))
}

#' رسم البواقي
plot_residuals <- function(model) {
  residuals_vec <- residuals(model)
  fitted_vals <- fitted(model)

  par(mfrow = c(2, 2))

  # 1. البواقي عبر الزمن
  plot(residuals_vec, type = "l", main = "البواقي عبر الزمن",
       xlab = "الزمن", ylab = "البواقي")
  abline(h = 0, col = "red", lty = 2)

  # 2. البواقي مقابل القيم الملائمة
  plot(fitted_vals, residuals_vec, main = "البواقي مقابل القيم الملائمة",
       xlab = "القيم الملائمة", ylab = "البواقي")
  abline(h = 0, col = "red", lty = 2)

  # 3. Q-Q Plot
  qqnorm(residuals_vec, main = "مخطط Q-Q للبواقي")
  qqline(residuals_vec, col = "red")

  # 4. Histogram
  hist(residuals_vec, main = "توزيع البواقي",
       xlab = "البواقي", col = "lightblue", border = "white")

  par(mfrow = c(1, 1))
}

# ============================================================================
# نهاية ملف الدوال المساعدة
# ============================================================================
