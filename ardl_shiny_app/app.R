# ============================================================================
# تطبيق ARDL Shiny - الاختيار الآلي للنماذج بمنهجية "التشخيص أولاً"
# ARDL Model Selection with "Diagnostics-First" Approach
# ============================================================================

# تحميل الدوال المساعدة
source("helpers.R")

# ============================================================================
# واجهة المستخدم (UI)
# ============================================================================

ui <- fluidPage(
  # إضافة دعم للغة العربية
  tags$head(
    tags$style(HTML("
      @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');

      body {
        font-family: 'Cairo', sans-serif;
        direction: rtl;
        text-align: right;
      }

      .navbar, .navbar-default {
        direction: rtl;
      }

      .form-group label {
        text-align: right;
        display: block;
      }

      .selectize-input {
        direction: rtl;
        text-align: right;
      }

      h1, h2, h3, h4 {
        font-family: 'Cairo', sans-serif;
        font-weight: 700;
      }

      .shiny-input-container {
        direction: rtl;
      }

      table {
        direction: ltr;
      }

      .alert-warning {
        background-color: #fff3cd;
        border-color: #ffc107;
        color: #856404;
        font-weight: bold;
      }

      .alert-success {
        background-color: #d4edda;
        border-color: #28a745;
        color: #155724;
        font-weight: bold;
      }

      .alert-danger {
        background-color: #f8d7da;
        border-color: #dc3545;
        color: #721c24;
        font-weight: bold;
      }
    "))
  ),

  # العنوان الرئيسي
  titlePanel(
    div(
      h1("نظام الاختيار الآلي لنماذج ARDL", align = "center",
         style = "color: #2c3e50; margin-bottom: 10px;"),
      h4("منهجية التشخيص أولاً - السلامة التشخيصية قبل معايير المعلومات",
         align = "center", style = "color: #7f8c8d;")
    )
  ),

  hr(),

  # التخطيط الرئيسي
  sidebarLayout(

    # ========================================================================
    # الشريط الجانبي - عناصر التحكم
    # ========================================================================
    sidebarPanel(
      width = 3,

      h3("إعدادات التطبيق", style = "color: #2980b9;"),

      # 1. استيراد البيانات
      h4("1. استيراد البيانات"),

      fileInput("data_file", "اختر ملف البيانات (CSV أو Excel)",
                accept = c(".csv", ".xlsx")),

      checkboxInput("use_demo_data", "استخدام بيانات تجريبية", FALSE),

      hr(),

      # 2. إعدادات الفهرس الزمني
      h4("2. الفهرس الزمني"),

      uiOutput("date_column_ui"),

      numericInput("start_year", "سنة البدء:", value = 2000, min = 1900, max = 2100),
      numericInput("start_period", "الفترة (1 للسنوي، 1-4 للربع سنوي، 1-12 للشهري):",
                   value = 1, min = 1, max = 12),

      selectInput("frequency", "التكرار:",
                  choices = c("سنوي" = "annual",
                             "ربع سنوي" = "quarterly",
                             "شهري" = "monthly"),
                  selected = "annual"),

      hr(),

      # 3. اختيار المتغيرات
      h4("3. تحديد المتغيرات"),

      uiOutput("dep_var_ui"),
      uiOutput("indep_vars_ui"),

      hr(),

      # 4. إعدادات النموذج
      h4("4. إعدادات ARDL"),

      numericInput("max_p", "الحد الأقصى لتأخيرات المتغير التابع (p):",
                   value = 4, min = 0, max = 8),

      numericInput("max_q", "الحد الأقصى لتأخيرات المتغيرات المستقلة (q):",
                   value = 4, min = 0, max = 8),

      selectInput("deterministics_mode", "اختيار المكونات الحتمية:",
                  choices = c("تلقائي" = "auto",
                             "ثابت فقط" = "const",
                             "اتجاه فقط" = "trend",
                             "ثابت + اتجاه" = "both",
                             "بدون" = "none"),
                  selected = "auto"),

      hr(),

      # 5. إعدادات الاختبارات
      h4("5. إعدادات الاختبارات"),

      selectInput("alpha", "مستوى المعنوية (α):",
                  choices = c("1%" = 0.01, "5%" = 0.05, "10%" = 0.10),
                  selected = 0.05),

      numericInput("bg_order", "رتبة اختبار Breusch-Godfrey:",
                   value = 2, min = 1, max = 8),

      checkboxInput("parallel_processing", "استخدام المعالجة المتوازية", TRUE),

      numericInput("n_cores", "عدد الأنوية:",
                   value = 2, min = 1, max = detectCores()),

      hr(),

      # 6. زر التشغيل
      actionButton("run_analysis", "تشغيل التحليل",
                   class = "btn-primary btn-lg btn-block",
                   style = "background-color: #27ae60; border: none; font-size: 18px; font-weight: bold;"),

      br(),

      # معلومات الحالة
      uiOutput("status_info")

    ),

    # ========================================================================
    # اللوحة الرئيسية - التبويبات
    # ========================================================================
    mainPanel(
      width = 9,

      tabsetPanel(
        id = "main_tabs",
        type = "tabs",

        # ====================================================================
        # التبويب 1: البيانات
        # ====================================================================
        tabPanel(
          "البيانات",
          icon = icon("table"),

          br(),
          h3("معاينة البيانات والإحصاءات الوصفية"),

          fluidRow(
            column(6,
                   h4("معلومات أساسية"),
                   verbatimTextOutput("data_info")
            ),
            column(6,
                   h4("الإحصاءات الوصفية"),
                   DTOutput("data_summary")
            )
          ),

          hr(),

          h4("معاينة البيانات الأولية"),
          DTOutput("data_preview"),

          hr(),

          h4("رسوم السلاسل الزمنية"),
          plotOutput("series_plots", height = "600px")
        ),

        # ====================================================================
        # التبويب 2: اختبارات الاستقرارية
        # ====================================================================
        tabPanel(
          "الاستقرارية",
          icon = icon("chart-line"),

          br(),
          h3("اختبارات جذر الوحدة (Unit Root Tests)"),

          fluidRow(
            column(12,
                   h4("ملخص نتائج الاختبارات"),
                   DTOutput("stationarity_summary")
            )
          ),

          hr(),

          h4("تفاصيل الاختبارات لكل متغير"),
          uiOutput("stationarity_details_ui"),

          hr(),

          h4("رسوم ACF و PACF"),
          plotOutput("acf_pacf_plots", height = "800px")
        ),

        # ====================================================================
        # التبويب 3: مساحة البحث
        # ====================================================================
        tabPanel(
          "مساحة البحث",
          icon = icon("search"),

          br(),
          h3("إعدادات البحث الآلي وتقدم التنفيذ"),

          fluidRow(
            column(4,
                   wellPanel(
                     h4("معلومات البحث"),
                     verbatimTextOutput("search_info")
                   )
            ),
            column(8,
                   h4("شريط التقدم"),
                   uiOutput("progress_bar"),
                   br(),
                   verbatimTextOutput("progress_text")
            )
          ),

          hr(),

          h4("النماذج المستبعدة (لم تجتاز المرشّحات)"),
          DTOutput("excluded_models_table")
        ),

        # ====================================================================
        # التبويب 4: مقارنة النماذج
        # ====================================================================
        tabPanel(
          "مقارنة النماذج",
          icon = icon("balance-scale"),

          br(),
          h3("أفضل النماذج المرشحة"),

          fluidRow(
            column(12,
                   numericInput("n_top_models", "عدد النماذج المعروضة:",
                               value = 5, min = 1, max = 20),

                   checkboxInput("show_all_models", "عرض جميع النماذج الناجحة", FALSE)
            )
          ),

          hr(),

          h4("جدول المقارنة الشامل"),
          DTOutput("comparison_table"),

          hr(),

          h4("رسم بياني للنقاط"),
          plotOutput("scores_plot", height = "400px")
        ),

        # ====================================================================
        # التبويب 5: النموذج الأفضل
        # ====================================================================
        tabPanel(
          "النموذج الأفضل",
          icon = icon("star"),

          br(),
          h3("تفاصيل النموذج المختار"),

          uiOutput("best_model_alert"),

          fluidRow(
            column(6,
                   wellPanel(
                     h4("معلومات النموذج"),
                     verbatimTextOutput("best_model_info")
                   )
            ),
            column(6,
                   wellPanel(
                     h4("معايير التقييم"),
                     verbatimTextOutput("best_model_criteria")
                   )
            )
          ),

          hr(),

          h4("معاملات الأجل الطويل (Long-Run)"),
          DTOutput("lr_coefficients"),

          hr(),

          h4("معاملات الأجل القصير (Short-Run) و ECT"),
          DTOutput("sr_coefficients"),

          hr(),

          h4("ملخص النموذج الكامل"),
          verbatimTextOutput("best_model_summary")
        ),

        # ====================================================================
        # التبويب 6: التشخيصات
        # ====================================================================
        tabPanel(
          "التشخيصات",
          icon = icon("stethoscope"),

          br(),
          h3("الاختبارات التشخيصية الشاملة"),

          fluidRow(
            column(6,
                   wellPanel(
                     h4("نتائج الاختبارات"),
                     DTOutput("diagnostics_tests")
                   )
            ),
            column(6,
                   wellPanel(
                     h4("Bounds Test"),
                     verbatimTextOutput("bounds_test_result")
                   )
            )
          ),

          hr(),

          h4("رسوم البواقي"),
          plotOutput("residuals_plots", height = "600px"),

          hr(),

          h4("اختبارات الاستقرار (CUSUM & CUSUMSQ)"),
          plotOutput("stability_plots", height = "500px"),

          hr(),

          h4("القيم الفعلية مقابل القيم الملائمة"),
          plotOutput("fitted_vs_actual", height = "400px")
        ),

        # ====================================================================
        # التبويب 7: التصدير
        # ====================================================================
        tabPanel(
          "التصدير",
          icon = icon("download"),

          br(),
          h3("تصدير النتائج والتقارير"),

          fluidRow(
            column(4,
                   wellPanel(
                     h4("تصدير الجداول"),
                     downloadButton("download_comparison_csv", "جدول المقارنة (CSV)",
                                   class = "btn-block"),
                     br(),
                     downloadButton("download_best_model_xlsx", "النموذج الأفضل (Excel)",
                                   class = "btn-block")
                   )
            ),
            column(4,
                   wellPanel(
                     h4("تصدير الرسوم"),
                     downloadButton("download_plots_pdf", "جميع الرسوم (PDF)",
                                   class = "btn-block"),
                     br(),
                     downloadButton("download_diagnostics_png", "التشخيصات (PNG)",
                                   class = "btn-block")
                   )
            ),
            column(4,
                   wellPanel(
                     h4("التقارير الشاملة"),
                     downloadButton("download_report_word", "تقرير Word",
                                   class = "btn-block"),
                     br(),
                     downloadButton("download_r_code", "كود R للإعادة",
                                   class = "btn-block")
                   )
            )
          ),

          hr(),

          h4("معلومات الجلسة (Session Info)"),
          verbatimTextOutput("session_info")
        )
      )
    )
  )
)

# ============================================================================
# منطق الخادم (Server)
# ============================================================================

server <- function(input, output, session) {

  # ==========================================================================
  # المتغيرات التفاعلية (Reactive Values)
  # ==========================================================================

  rv <- reactiveValues(
    data_raw = NULL,
    data_ts = NULL,
    variables = NULL,
    stationarity_results = NULL,
    search_grid = NULL,
    all_models = NULL,
    ranked_models = NULL,
    best_model = NULL,
    progress_value = 0,
    progress_message = "",
    excluded_models = list()
  )

  # ==========================================================================
  # 1. استيراد البيانات
  # ==========================================================================

  observeEvent(input$use_demo_data, {
    if (input$use_demo_data) {
      rv$data_raw <- generate_synthetic_data(n = 100, seed = 123)
      showNotification("تم تحميل البيانات التجريبية بنجاح", type = "message")
    }
  })

  observeEvent(input$data_file, {
    req(input$data_file)

    file_ext <- tools::file_ext(input$data_file$name)

    rv$data_raw <- tryCatch({
      if (file_ext == "csv") {
        data_ingest(input$data_file$datapath, "csv")
      } else if (file_ext %in% c("xlsx", "xls")) {
        data_ingest(input$data_file$datapath, "xlsx")
      } else {
        showNotification("نوع الملف غير مدعوم", type = "error")
        NULL
      }
    }, error = function(e) {
      showNotification(paste("خطأ:", e$message), type = "error")
      NULL
    })

    if (!is.null(rv$data_raw)) {
      showNotification("تم تحميل البيانات بنجاح", type = "message")
    }
  })

  # واجهة اختيار عمود التاريخ
  output$date_column_ui <- renderUI({
    req(rv$data_raw)
    selectInput("date_column", "عمود التاريخ (اختياري):",
                choices = c("بدون" = "", names(rv$data_raw)),
                selected = "")
  })

  # واجهة اختيار المتغير التابع
  output$dep_var_ui <- renderUI({
    req(rv$data_raw)
    numeric_vars <- names(rv$data_raw)[sapply(rv$data_raw, is.numeric)]
    selectInput("dep_var", "المتغير التابع:",
                choices = numeric_vars)
  })

  # واجهة اختيار المتغيرات المستقلة
  output$indep_vars_ui <- renderUI({
    req(rv$data_raw, input$dep_var)
    numeric_vars <- names(rv$data_raw)[sapply(rv$data_raw, is.numeric)]
    available_vars <- setdiff(numeric_vars, input$dep_var)

    checkboxGroupInput("indep_vars", "المتغيرات المستقلة:",
                       choices = available_vars,
                       selected = available_vars[1])
  })

  # ==========================================================================
  # 2. معاينة البيانات
  # ==========================================================================

  output$data_info <- renderPrint({
    req(rv$data_raw)
    cat("عدد المشاهدات:", nrow(rv$data_raw), "\n")
    cat("عدد المتغيرات:", ncol(rv$data_raw), "\n")
    cat("أسماء المتغيرات:", paste(names(rv$data_raw), collapse = ", "), "\n")
  })

  output$data_summary <- renderDT({
    req(rv$data_raw)
    numeric_data <- rv$data_raw[, sapply(rv$data_raw, is.numeric), drop = FALSE]

    summary_df <- data.frame(
      Variable = names(numeric_data),
      Mean = sapply(numeric_data, mean, na.rm = TRUE),
      SD = sapply(numeric_data, sd, na.rm = TRUE),
      Min = sapply(numeric_data, min, na.rm = TRUE),
      Max = sapply(numeric_data, max, na.rm = TRUE)
    )

    datatable(summary_df, options = list(pageLength = 10, dom = 't'),
              rownames = FALSE) %>%
      formatRound(columns = 2:5, digits = 3)
  })

  output$data_preview <- renderDT({
    req(rv$data_raw)
    datatable(head(rv$data_raw, 20), options = list(pageLength = 10, scrollX = TRUE))
  })

  output$series_plots <- renderPlot({
    req(rv$data_raw)

    numeric_vars <- names(rv$data_raw)[sapply(rv$data_raw, is.numeric)]
    n_vars <- length(numeric_vars)

    par(mfrow = c(ceiling(n_vars/2), 2), mar = c(4, 4, 2, 1))

    for (var in numeric_vars) {
      plot(rv$data_raw[[var]], type = "l", main = var,
           xlab = "المشاهدة", ylab = "القيمة", col = "steelblue", lwd = 2)
      grid()
    }
  })

  # ==========================================================================
  # 3. التحليل الرئيسي
  # ==========================================================================

  observeEvent(input$run_analysis, {

    # التحقق من المدخلات
    req(rv$data_raw, input$dep_var, input$indep_vars)

    # إعادة تعيين النتائج
    rv$stationarity_results <- NULL
    rv$all_models <- NULL
    rv$ranked_models <- NULL
    rv$best_model <- NULL
    rv$excluded_models <- list()

    withProgress(message = 'جارٍ التحليل...', value = 0, {

      # الخطوة 1: إنشاء الفهرس الزمني
      incProgress(0.1, detail = "إنشاء الفهرس الزمني")

      rv$data_ts <- time_index_resolver(
        rv$data_raw,
        date_col = if (input$date_column != "") input$date_column else NULL,
        start_date = c(input$start_year, input$start_period),
        frequency = input$frequency
      )

      # الخطوة 2: اختبارات الاستقرارية
      incProgress(0.2, detail = "اختبارات الاستقرارية")

      all_vars <- c(input$dep_var, input$indep_vars)
      stationarity_list <- list()

      for (var in all_vars) {
        stationarity_list[[var]] <- unit_root_tests(
          series = as.numeric(rv$data_ts[, var]),
          var_name = var,
          alpha = as.numeric(input$alpha)
        )
      }

      rv$stationarity_results <- stationarity_list

      # الخطوة 3: بناء شبكة البحث
      incProgress(0.3, detail = "بناء شبكة النماذج")

      deterministics <- if (input$deterministics_mode == "auto") {
        c("const", "trend", "both", "none")
      } else {
        input$deterministics_mode
      }

      rv$search_grid <- build_search_grid(
        dep_var = input$dep_var,
        indep_vars = input$indep_vars,
        max_p = input$max_p,
        max_q = input$max_q,
        deterministics = deterministics
      )

      n_models <- nrow(rv$search_grid)
      showNotification(paste("سيتم تقدير", n_models, "نموذج"), type = "message")

      # الخطوة 4: تقدير جميع النماذج
      incProgress(0.4, detail = paste("تقدير", n_models, "نموذج"))

      models_results <- list()

      for (i in 1:min(n_models, 500)) {  # حد أقصى 500 نموذج لتجنب التعليق

        model_spec <- rv$search_grid[i, ]

        # استخراج p و q's
        p <- model_spec$p
        q_vec <- as.numeric(model_spec[input$indep_vars])

        # تقدير النموذج
        model <- fit_ardl_candidate(
          data = as.data.frame(rv$data_ts),
          dep_var = input$dep_var,
          indep_vars = input$indep_vars,
          p = p,
          q_vector = q_vec,
          deterministic = model_spec$deterministic
        )

        if (is.null(model)) {
          rv$excluded_models[[length(rv$excluded_models) + 1]] <- list(
            model_id = i,
            spec = model_spec,
            reason = "فشل التقدير"
          )
          next
        }

        # التشخيصات
        diag <- diagnostics_bundle(model, alpha = as.numeric(input$alpha),
                                   bg_order = input$bg_order)

        if (diag$failed) {
          rv$excluded_models[[length(rv$excluded_models) + 1]] <- list(
            model_id = i,
            spec = model_spec,
            reason = diag$reason
          )
          next
        }

        # ECT
        ect <- ect_extract(model, alpha = as.numeric(input$alpha))

        # الاستقرار
        stability <- stability_tests(model, alpha = as.numeric(input$alpha))

        # Bounds Test
        case_num <- switch(model_spec$deterministic,
                          "none" = 1, "const" = 3, "trend" = 2, "both" = 5, 3)

        bounds <- bounds_test_wrapper(model, case_num = case_num,
                                      alpha = as.numeric(input$alpha))

        # التقييم
        aic_rank <- i / n_models  # مبسط
        sign_check <- list(matches = 0, conflicts = 0)  # يمكن توسيعه

        score_result <- score_model(
          model = model,
          diagnostics = diag,
          ect_result = ect,
          stability_result = stability,
          bounds_result = bounds,
          sign_check = sign_check,
          aic_rank = aic_rank,
          alpha = as.numeric(input$alpha)
        )

        if (score_result$score == 0) {
          rv$excluded_models[[length(rv$excluded_models) + 1]] <- list(
            model_id = i,
            spec = model_spec,
            reason = score_result$reason
          )
          next
        }

        # حفظ النتائج
        models_results[[length(models_results) + 1]] <- list(
          model_id = i,
          spec = model_spec,
          model = model,
          diagnostics = diag,
          ect = ect,
          stability = stability,
          bounds = bounds,
          score_result = score_result
        )

        # تحديث التقدم
        if (i %% 10 == 0) {
          incProgress(0.4 * i / n_models, detail = paste("تم تقدير", i, "من", n_models))
        }
      }

      rv$all_models <- models_results

      # الخطوة 5: الترتيب
      incProgress(0.9, detail = "ترتيب النماذج")

      ranking_result <- rank_and_summarize(models_results)
      rv$ranked_models <- ranking_result$top_models

      if (!is.null(rv$ranked_models) && length(rv$ranked_models) > 0) {
        rv$best_model <- rv$ranked_models[[1]]
        showNotification("اكتمل التحليل بنجاح!", type = "message")
      } else {
        showNotification("لم يجتاز أي نموذج المرشّحات الصارمة", type = "warning")
      }

      incProgress(1.0, detail = "اكتمل")
    })
  })

  # ==========================================================================
  # 4. عرض نتائج الاستقرارية
  # ==========================================================================

  output$stationarity_summary <- renderDT({
    req(rv$stationarity_results)

    summary_list <- list()

    for (var_name in names(rv$stationarity_results)) {
      result <- rv$stationarity_results[[var_name]]

      summary_list[[var_name]] <- data.frame(
        Variable = var_name,
        ADF_Drift = ifelse(!is.null(result$adf$drift),
                          result$adf$drift$decision, "N/A"),
        PP = ifelse(!is.null(result$pp),
                   result$pp$decision, "N/A"),
        KPSS_Level = ifelse(!is.null(result$kpss$level),
                           result$kpss$level$decision, "N/A"),
        stringsAsFactors = FALSE
      )
    }

    summary_df <- do.call(rbind, summary_list)
    datatable(summary_df, options = list(dom = 't'), rownames = FALSE)
  })

  output$acf_pacf_plots <- renderPlot({
    req(rv$data_ts, input$dep_var, input$indep_vars)

    all_vars <- c(input$dep_var, input$indep_vars)
    n_vars <- length(all_vars)

    par(mfrow = c(n_vars, 2), mar = c(3, 3, 2, 1))

    for (var in all_vars) {
      series <- as.numeric(rv$data_ts[, var])
      acf(series, main = paste("ACF -", var))
      pacf(series, main = paste("PACF -", var))
    }
  })

  # ==========================================================================
  # 5. مقارنة النماذج
  # ==========================================================================

  output$comparison_table <- renderDT({
    req(rv$ranked_models)

    n_show <- ifelse(input$show_all_models, length(rv$ranked_models),
                    min(input$n_top_models, length(rv$ranked_models)))

    comparison_list <- list()

    for (i in 1:n_show) {
      m <- rv$ranked_models[[i]]

      comparison_list[[i]] <- data.frame(
        Rank = i,
        Model_ID = m$model_id,
        p = m$spec$p,
        q_vars = paste(as.numeric(m$spec[input$indep_vars]), collapse = ","),
        Deterministic = m$spec$deterministic,
        BG_Pass = ifelse(m$diagnostics$bg$passed, "✓", "✗"),
        ECT_Pass = ifelse(m$ect$passed, "✓", "✗"),
        Stability = ifelse(m$stability$both_passed, "✓", "✗"),
        AIC = round(m$diagnostics$aic, 2),
        BIC = round(m$diagnostics$bic, 2),
        Score = m$score_result$score,
        stringsAsFactors = FALSE
      )
    }

    comparison_df <- do.call(rbind, comparison_list)

    datatable(comparison_df,
              options = list(pageLength = 10, scrollX = TRUE),
              rownames = FALSE) %>%
      formatStyle('BG_Pass',
                  backgroundColor = styleEqual(c("✓", "✗"),
                                              c('#d4edda', '#f8d7da'))) %>%
      formatStyle('ECT_Pass',
                  backgroundColor = styleEqual(c("✓", "✗"),
                                              c('#d4edda', '#f8d7da')))
  })

  output$scores_plot <- renderPlot({
    req(rv$ranked_models)

    n_show <- min(input$n_top_models, length(rv$ranked_models))
    scores <- sapply(rv$ranked_models[1:n_show], function(x) x$score_result$score)

    barplot(scores, names.arg = 1:n_show,
            main = "النقاط الكلية لأفضل النماذج",
            xlab = "رتبة النموذج", ylab = "النقاط",
            col = "steelblue", border = "white")
    grid(nx = NA, ny = NULL)
  })

  # ==========================================================================
  # 6. النموذج الأفضل
  # ==========================================================================

  output$best_model_alert <- renderUI({
    req(rv$best_model)

    div(
      class = "alert alert-success",
      h4(icon("check-circle"), " تم اختيار النموذج الأفضل بنجاح"),
      p("النموذج التالي حقق أعلى نقاط وفق معايير التشخيص أولاً")
    )
  })

  output$best_model_info <- renderPrint({
    req(rv$best_model)

    cat("رقم النموذج:", rv$best_model$model_id, "\n")
    cat("الترتيب: ARDL(", rv$best_model$spec$p, ", ",
        paste(as.numeric(rv$best_model$spec[input$indep_vars]), collapse = ", "),
        ")\n", sep = "")
    cat("المكونات الحتمية:", rv$best_model$spec$deterministic, "\n")
    cat("النقاط الكلية:", rv$best_model$score_result$score, "\n")
  })

  output$best_model_criteria <- renderPrint({
    req(rv$best_model)

    cat("AIC:", round(rv$best_model$diagnostics$aic, 4), "\n")
    cat("BIC:", round(rv$best_model$diagnostics$bic, 4), "\n")
    cat("Breusch-Godfrey p:", round(rv$best_model$diagnostics$bg$p_value, 4), "\n")
    cat("Breusch-Pagan p:", round(rv$best_model$diagnostics$bp$p_value, 4), "\n")
    cat("Jarque-Bera p:", round(rv$best_model$diagnostics$jb$p_value, 4), "\n")
  })

  output$lr_coefficients <- renderDT({
    req(rv$best_model)

    lr_result <- tryCatch({
      multipliers(rv$best_model$model, type = "lr")
    }, error = function(e) NULL)

    if (!is.null(lr_result)) {
      lr_df <- data.frame(
        Variable = rownames(lr_result),
        Estimate = lr_result$estimate,
        Std_Error = lr_result$std.error,
        t_value = lr_result$statistic,
        p_value = lr_result$p.value
      )

      datatable(lr_df, options = list(dom = 't'), rownames = FALSE) %>%
        formatRound(columns = 2:5, digits = 4) %>%
        formatStyle('p_value',
                   backgroundColor = styleInterval(0.05, c('#d4edda', 'white')))
    }
  })

  output$sr_coefficients <- renderDT({
    req(rv$best_model)

    ecm_model <- recm(rv$best_model$model, case = rv$best_model$model$case)
    coefs <- summary(ecm_model)$coefficients

    coefs_df <- data.frame(
      Variable = rownames(coefs),
      Estimate = coefs[, "Estimate"],
      Std_Error = coefs[, "Std. Error"],
      t_value = coefs[, "t value"],
      p_value = coefs[, "Pr(>|t|)"]
    )

    datatable(coefs_df, options = list(pageLength = 15, scrollX = TRUE),
              rownames = FALSE) %>%
      formatRound(columns = 2:5, digits = 4) %>%
      formatStyle('p_value',
                 backgroundColor = styleInterval(0.05, c('#d4edda', 'white')))
  })

  output$best_model_summary <- renderPrint({
    req(rv$best_model)
    summary(rv$best_model$model)
  })

  # ==========================================================================
  # 7. التشخيصات
  # ==========================================================================

  output$diagnostics_tests <- renderDT({
    req(rv$best_model)

    diag <- rv$best_model$diagnostics

    tests_df <- data.frame(
      Test = c("Breusch-Godfrey", "Breusch-Pagan", "ARCH", "Jarque-Bera"),
      Statistic = c(diag$bg$statistic, diag$bp$statistic,
                   ifelse(!is.null(diag$arch$statistic), diag$arch$statistic, NA),
                   diag$jb$statistic),
      p_value = c(diag$bg$p_value, diag$bp$p_value,
                 ifelse(!is.null(diag$arch$p_value), diag$arch$p_value, NA),
                 diag$jb$p_value),
      Result = c(ifelse(diag$bg$passed, "ناجح", "فاشل"),
                ifelse(diag$bp$passed, "ناجح", "فاشل"),
                ifelse(diag$arch$passed, "ناجح", "فاشل"),
                ifelse(diag$jb$passed, "ناجح", "فاشل"))
    )

    datatable(tests_df, options = list(dom = 't'), rownames = FALSE) %>%
      formatRound(columns = 2:3, digits = 4) %>%
      formatStyle('Result',
                 backgroundColor = styleEqual(c("ناجح", "فاشل"),
                                             c('#d4edda', '#f8d7da')))
  })

  output$bounds_test_result <- renderPrint({
    req(rv$best_model)

    bounds <- rv$best_model$bounds

    if (!is.null(bounds) && !bounds$failed) {
      cat("F-Statistic:", round(bounds$f_statistic, 4), "\n")
      cat("Lower Bound:", round(bounds$lower_bound, 4), "\n")
      cat("Upper Bound:", round(bounds$upper_bound, 4), "\n")
      cat("القرار:", bounds$decision, "\n")
    } else {
      cat("فشل اختبار Bounds Test\n")
    }
  })

  output$residuals_plots <- renderPlot({
    req(rv$best_model)
    plot_residuals(rv$best_model$model)
  })

  output$stability_plots <- renderPlot({
    req(rv$best_model)

    ols_model <- lm(rv$best_model$model$formula, data = rv$best_model$model$model)

    par(mfrow = c(1, 2))

    # CUSUM
    cusum <- efp(ols_model, type = "Rec-CUSUM")
    plot(cusum, main = "CUSUM Test")

    # CUSUMSQ
    cusumsq <- efp(ols_model, type = "OLS-CUSUM")
    plot(cusumsq, main = "CUSUMSQ Test")
  })

  output$fitted_vs_actual <- renderPlot({
    req(rv$best_model)

    actual <- rv$best_model$model$model[, 1]
    fitted <- fitted(rv$best_model$model)

    plot(actual, type = "l", col = "black", lwd = 2,
         main = "القيم الفعلية مقابل القيم الملائمة",
         xlab = "المشاهدة", ylab = "القيمة")
    lines(fitted, col = "red", lwd = 2, lty = 2)
    legend("topleft", legend = c("فعلي", "ملائم"),
           col = c("black", "red"), lty = c(1, 2), lwd = 2)
    grid()
  })

  # ==========================================================================
  # 8. معلومات الحالة
  # ==========================================================================

  output$status_info <- renderUI({
    if (!is.null(rv$best_model)) {
      div(
        class = "alert alert-success",
        p(strong("الحالة:"), "تم التحليل بنجاح"),
        p(strong("عدد النماذج الناجحة:"), length(rv$ranked_models)),
        p(strong("النموذج الأفضل:"), paste0("ARDL(", rv$best_model$spec$p, ")"))
      )
    } else if (!is.null(rv$all_models)) {
      div(
        class = "alert alert-warning",
        p("لا توجد نماذج ناجحة")
      )
    } else {
      div(
        class = "alert alert-info",
        p("في انتظار بدء التحليل")
      )
    }
  })

  output$session_info <- renderPrint({
    sessionInfo()
  })

  # ==========================================================================
  # 9. التصدير (مبسط)
  # ==========================================================================

  output$download_comparison_csv <- downloadHandler(
    filename = function() {
      paste0("ardl_comparison_", Sys.Date(), ".csv")
    },
    content = function(file) {
      # منطق بسيط للتصدير
      write.csv(data.frame(Message = "قريباً"), file, row.names = FALSE)
    }
  )

  # يمكن إضافة المزيد من التصديرات بشكل مشابه

}

# ============================================================================
# تشغيل التطبيق
# ============================================================================

shinyApp(ui = ui, server = server)
