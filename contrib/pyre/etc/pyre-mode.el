;
; michael a.g. aïvázis
; orthologue
; (c) 1998-2019 all rights reserved
;

;; requirements
(require 'font-lock)
(require 'easymenu)

;; customization
(defconst pyre-mode-version "1.0"
  "The version of `pyre-mode'.")

(defgroup pyre nil
  "The major mode for pyre configuration files"
  :group 'tools)

(defcustom pyre-tab-width tab-width
  "The number of spaces to use when indenting"
  :type 'integer
  :group 'pyre
  :safe 'integerp)

;; key bindings
(defvar pyre-mode-map
  (let ((map (make-sparse-keymap)))
    ;; tabs
    (define-key map [remap newline-and-indent] 'pyre-indent-line)
    (define-key map (kbd "<backtab>") 'pyre-indent-shift-left)
    ;; moving regions around
    (define-key map (kbd "C-c C-<") 'pyre-indent-shift-left)
    (define-key map (kbd "C-c C->") 'pyre-indent-shift-right)
    ;; done
    map)
  "Keymap for the pyre major mode.")

;; implementation
(defun pyre-version ()
  "Show the `pyre-mode' version in the echo area."
  (interactive)
  (message (concat "pyre-mode version " pyre-mode-version)))

;; indentation
(defun pyre-previous-indentation ()
  "Gets indentation of previous line"
  (save-excursion
    (previous-line)
    (current-indentation)))

(defun pyre-max-indent ()
  "Calculates max indentation"
  (+ (pyre-previous-indentation) pyre-tab-width))

(defun pyre-empty-line-p ()
  "If line is completely empty"
  (= (point-at-bol) (point-at-eol)))

(defun pyre-point-to-bot ()
  "Moves point to beginning of text"
  (beginning-of-line-text))

(defun pyre-do-indent-line ()
  "Performs line indentation"
  ;;if we are not tabbed out past max indent
  (if (pyre-empty-line-p)
      (indent-to (pyre-max-indent))
    (if (< (current-indentation) (pyre-max-indent))
        (indent-to (+ (current-indentation) pyre-tab-width))
      ;; if at max indent move text to beginning of line
      (progn
        (beginning-of-line)
        (delete-horizontal-space)))))

(defun pyre-indent-line ()
  "Indents current line"
  (interactive)
  (if mark-active
      (pyre-indent-region)
    (if (pyre-at-bot-p)
        (pyre-do-indent-line)
      ;; instead of adjusting indent, move point to text
      (pyre-point-to-bot))))

(defun pyre-at-bol-p ()
  "If point is at beginning of line"
  (interactive)
  (= (point) (point-at-bol)))

(defun pyre-at-bot-p ()
  "If point is at beginning of text"
  (= (point) (+ (current-indentation) (point-at-bol))))

(defun pyre-print-line-number ()
  "Prints line number"
  (pyre-print-num (point)))

(defun pyre-print-num (arg)
  "Prints line number"
  (message (number-to-string arg)))

(defun pyre-indent-to (num)
  "Force indentation to level including those below current level"
  (save-excursion
    (beginning-of-line)
    (delete-horizontal-space)
    (indent-to num)))

(defun pyre-move-region (begin end prog)
  "Moves left if dir is null, otherwise right. prog is '+ or '-"
  (save-excursion
    (let (first-indent indent-diff)
      (goto-char begin)
      (setq first-indent (current-indentation))
      (pyre-indent-to (funcall prog first-indent pyre-tab-width))
      (setq indent-diff (- (current-indentation) first-indent))
      ;; move other lines based on movement of first line
      (while (< (point) end)
        (forward-line 1)
        (if (< (point) end)
            (pyre-indent-to (+ (current-indentation) indent-diff)))))))

(defun pyre-indent-region (begin end)
  "Indents the selected region"
  (interactive)
  (pyre-move-region begin end '+))

(defun pyre-dedent-line ()
  "Dedents the current line"
  (interactive)
  (if mark-active
      (pyre-move-region (region-beginning) (region-end) '-)
    (if (pyre-at-bol-p)
        (progn
          (delete-horizontal-space)
          (indent-to (pyre-max-indent)))
      (let ((ci (current-indentation)))
        (beginning-of-line)
        (delete-horizontal-space)
        (indent-to (- ci pyre-tab-width))))))

;; menu
(easy-menu-define pyre-mode-menu pyre-mode-map
  "Menu for pyre mode"
  '("pyre"
    "---"
    ["Version" pyre-version]
    ))

;; syntax
;; comments
(defvar pyre-comment-regexp ";.*")
;; sections
(defvar pyre-section-regexp "^\s*\\(.+\\)\s*:")
(defvar pyre-conditional-section-regexp "^\s*\\(.+\\)\s*\\(#\\)\s*\\(.+\\)\s*:")
;; lhs of assignments
(defvar pyre-assignment-regexp "^\s*\\(.*\\)\s*=")
;; rhs of assignments
(defvar pyre-value-regexp "=\s*\\(.*\\)$")
;; interpolations
(defvar pyre-interpolation-regexp "{[^{].*}")
;; lines with no special markings are strings
(defvar pyre-continuation-regexp "^\s+\\(.*\\)$")
;; syntax highlighting
(defvar pyre-font-lock-table
  `(
    (, pyre-comment-regexp . font-lock-comment-face)
    (, pyre-conditional-section-regexp 1 font-lock-type-face)
    (, pyre-conditional-section-regexp 2 font-lock-builtin-face)
    (, pyre-conditional-section-regexp 3 font-lock-function-name-face)
    (, pyre-section-regexp . font-lock-type-face)
    (, pyre-interpolation-regexp . font-lock-variable-name-face)
    (, pyre-assignment-regexp . font-lock-variable-name-face)
    (, pyre-value-regexp 1 font-lock-string-face)
    (, pyre-continuation-regexp 1 font-lock-string-face)
    ;(, pyre-xxx-regexp . font-lock-xxx-face)
    ))

;; main
(define-derived-mode pyre-mode fundamental-mode "pyre"
  "Major mode for pyre configuration files"
  ; clean slate
  (kill-all-local-variables)
  ; identify this mode
  (setq mode-name "Pyre")
  (setq major-mode 'pyre-mode)

  ; indentation
  (make-local-variable 'tab-width)
  (make-local-variable 'pyre-tab-width)
  (setq pyre-tab-width 2)

  (make-local-variable 'indent-line-function)
  (setq indent-line-function 'pyre-indent-line)

  (make-local-variable 'indent-region-function)
  (setq indent-region-function 'pyre-indent-region)


  ;; keymap
  (use-local-map pyre-mode-map)

  ; install the font-lock table
  (setq font-lock-defaults '(pyre-font-lock-table))
  )

;; what we do
(provide 'pyre-mode)

;;;###autoload
(add-to-list 'auto-mode-alist '("\\.pfg$" . pyre-mode))

; end of file
