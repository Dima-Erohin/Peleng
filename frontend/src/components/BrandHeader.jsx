import React from "react";

function MetaChip({ label, value }) {
  return <span className="meta-chip"><span>{label}</span><b>{value}</b></span>;
}

export function BrandHeader({ connected, status, mode, detector, children }) {
  return (
    <header className="topbar">
      <a className="brand" href={import.meta.env.DEV ? import.meta.env.BASE_URL : "/"} aria-label="Peleng — главная">
        <img className="brand-logo" src="/static/build/brand/peleng-logo-red.svg" alt="ПЕЛЕНГ" />
        <span className="brand-caption">3D TRACKING</span>
      </a>
      {children}
      <div className="topbar-center">
        <span className={`live-indicator ${connected ? "" : "offline"}`}><i />{status}</span>
      </div>
      <div className="topbar-meta">
        <MetaChip label="Режим" value={mode} />
        <MetaChip label="Детектор" value={detector} />
      </div>
    </header>
  );
}
