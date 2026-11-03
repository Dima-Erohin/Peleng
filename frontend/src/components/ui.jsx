import React from "react";

export function Panel({ title, children, className = "", meta, actions }) {
  return (
    <section className={`panel-section ${className}`}>
      <header className="panel-heading">
        <h2>{title}</h2>
        {meta && <span className="panel-meta">{meta}</span>}
        {actions}
      </header>
      {children}
    </section>
  );
}

export function Button({ variant = "secondary", size = "default", className = "", children, ...props }) {
  const classes = ["button", `button-${variant}`, size !== "default" && `button-${size}`, className]
    .filter(Boolean)
    .join(" ");
  return <button className={classes} {...props}>{children}</button>;
}

export function StatusBadge({ children, tone = "neutral" }) {
  return <span className={`status-badge status-${tone}`}><i />{children}</span>;
}
