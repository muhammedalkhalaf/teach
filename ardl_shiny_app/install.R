# ============================================================================
# سكربت تثبيت الحزم المطلوبة لتطبيق ARDL Shiny
# Install Required Packages for ARDL Shiny App
# ============================================================================

cat("تثبيت الحزم المطلوبة لتطبيق ARDL Shiny...\n")
cat("Installing required packages for ARDL Shiny App...\n\n")

# قائمة الحزم المطلوبة
required_packages <- c(
  # Shiny و واجهة المستخدم
  "shiny",
  "DT",
  "shinythemes",
  "shinyWidgets",

  # قراءة وكتابة البيانات
  "readr",
  "readxl",
  "writexl",
  "data.table",

  # السلاسل الزمنية
  "zoo",
  "xts",
  "forecast",
  "tseries",

  # نمذجة ARDL
  "ARDL",
  "dynamac",

  # الاختبارات الإحصائية
  "lmtest",
  "sandwich",
  "urca",
  "car",
  "strucchange",
  "FinTS",  # لاختبار ARCH

  # الرسوم البيانية
  "ggplot2",
  "gridExtra",
  "plotly",

  # المعالجة المتوازية
  "parallel",
  "doParallel",

  # التقارير
  "rmarkdown",
  "knitr",
  "officer",     # لتقارير Word
  "flextable",   # لجداول في Word
  "kableExtra",

  # أدوات عامة
  "dplyr",
  "tidyr",
  "tibble"
)

# دالة للتحقق من وجود الحزمة وتثبيتها إن لزم
install_if_missing <- function(package) {
  if (!require(package, character.only = TRUE, quietly = TRUE)) {
    cat(paste("تثبيت الحزمة:", package, "...\n"))
    cat(paste("Installing package:", package, "...\n"))

    tryCatch({
      install.packages(package, dependencies = TRUE, repos = "https://cran.r-project.org")
      cat(paste("✓ تم تثبيت", package, "بنجاح\n\n"))
    }, error = function(e) {
      cat(paste("✗ فشل تثبيت", package, "\n"))
      cat(paste("Error:", e$message, "\n\n"))
    })
  } else {
    cat(paste("✓", package, "موجودة بالفعل\n"))
  }
}

# تثبيت كل الحزم
for (pkg in required_packages) {
  install_if_missing(pkg)
}

cat("\n" , rep("=", 70), "\n")
cat("اكتمل تثبيت الحزم!\n")
cat("Installation complete!\n")
cat(rep("=", 70), "\n\n")

# عرض معلومات R
cat("معلومات النظام / System Information:\n")
cat("إصدار R / R Version:", R.version.string, "\n")
cat("النظام / Platform:", R.version$platform, "\n")

cat("\n✓ يمكنك الآن تشغيل التطبيق باستخدام:\n")
cat("  You can now run the app using:\n\n")
cat("  shiny::runApp('app.R')\n\n")

# التحقق من تحميل الحزم الأساسية
cat("التحقق من تحميل الحزم الأساسية...\n")
cat("Checking critical packages...\n\n")

critical_packages <- c("shiny", "ARDL", "lmtest", "urca", "strucchange")
all_loaded <- TRUE

for (pkg in critical_packages) {
  loaded <- require(pkg, character.only = TRUE, quietly = TRUE)
  status <- ifelse(loaded, "✓ نجح", "✗ فشل")
  cat(paste(status, "-", pkg, "\n"))
  if (!loaded) all_loaded <- FALSE
}

if (all_loaded) {
  cat("\n✓✓✓ جميع الحزم الأساسية جاهزة!\n")
  cat("    All critical packages are ready!\n")
} else {
  cat("\n✗✗✗ بعض الحزم لم يتم تحميلها. يرجى التحقق من الأخطاء أعلاه.\n")
  cat("    Some packages failed to load. Please check errors above.\n")
}
