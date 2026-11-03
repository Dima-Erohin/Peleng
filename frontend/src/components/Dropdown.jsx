import React, { useEffect, useId, useRef, useState } from "react";

export function Dropdown({ label, disabled = false, trigger, children, className = "", onOpenChange, popupRole = "listbox" }) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef(null);
  const triggerRef = useRef(null);
  const menuId = useId();

  useEffect(() => {
    onOpenChange?.(open);
  }, [open, onOpenChange]);

  useEffect(() => {
    if (!open) return undefined;
    const onPointerDown = (event) => {
      if (!rootRef.current?.contains(event.target)) setOpen(false);
    };
    const onKeyDown = (event) => {
      if (event.key === "Escape") {
        setOpen(false);
        triggerRef.current?.focus();
      }
    };
    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  return (
    <div className={`dropdown ${className}`} ref={rootRef}>
      {React.cloneElement(trigger, {
        "aria-haspopup": popupRole,
        "aria-expanded": open,
        "aria-controls": menuId,
        disabled,
        ref: triggerRef,
        onClick: (event) => {
          trigger.props.onClick?.(event);
          if (!disabled) setOpen((current) => !current);
        },
        onKeyDown: (event) => {
          trigger.props.onKeyDown?.(event);
          if (event.defaultPrevented) return;
          if (event.key === "ArrowDown" || event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            if (!disabled) setOpen(true);
          }
        },
      })}
      {open && (
        <div className="dropdown-menu" id={menuId} role={popupRole} aria-label={label}>
          {typeof children === "function" ? children(() => setOpen(false)) : children}
        </div>
      )}
    </div>
  );
}

export function Select({ label, value, options, onChange, disabled = false }) {
  const selected = options.find((option) => option.value === value) || options[0];
  const optionRefs = useRef([]);
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    if (menuOpen) {
      const selectedIndex = options.findIndex((option) => option.value === value);
      optionRefs.current[Math.max(selectedIndex, 0)]?.focus();
    }
  }, [menuOpen]);

  const onMenuKeyDown = (event) => {
    if (!options.length) return;
    const activeIndex = optionRefs.current.indexOf(document.activeElement);
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      const direction = event.key === "ArrowDown" ? 1 : -1;
      const start = activeIndex < 0 ? 0 : activeIndex;
      const next = (start + direction + options.length) % options.length;
      optionRefs.current[next]?.focus();
    } else if (event.key === "Home" || event.key === "End") {
      event.preventDefault();
      optionRefs.current[event.key === "Home" ? 0 : options.length - 1]?.focus();
    }
  };

  return (
    <Dropdown
      label={label}
      disabled={disabled}
      className="select-dropdown"
      onOpenChange={setMenuOpen}
      trigger={(
        <button type="button" className="select-trigger" aria-label={label}>
          <span>{selected?.label || "Выберите значение"}</span>
          <svg viewBox="0 0 16 16" aria-hidden="true"><path d="m3.5 6 4.5 4 4.5-4" /></svg>
        </button>
      )}
    >
      {(close) => {
        return (
          <div className="select-options" onKeyDown={onMenuKeyDown}>
            {options.map((option, index) => (
              <button
                type="button"
                role="option"
                aria-selected={option.value === value}
                className={`select-option ${option.value === value ? "is-selected" : ""}`}
                key={option.value}
                ref={(element) => { optionRefs.current[index] = element; }}
                onClick={() => {
                  onChange(option.value);
                  close();
                }}
              >
                <span>{option.label}</span>
                {option.value === value && <span className="select-check" aria-hidden="true">✓</span>}
              </button>
            ))}
          </div>
        );
      }}
    </Dropdown>
  );
}
