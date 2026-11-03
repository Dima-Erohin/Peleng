import React from "react";

export default function CameraPreviews({ cameras, showBoard }) {
  const onlineCount = cameras.filter((camera) => camera.online).length;
  return (
    <section className="preview-strip">
      <header className="strip-heading">
        <h2>Потоки камер</h2>
        <span>{onlineCount} из {cameras.length} на связи</span>
      </header>
      {cameras.length ? (
        <div id="previews" className="preview-list">
          {cameras.map((camera) => (
            <figure className="preview-card" key={camera.id}>
              <img
                alt={`Кадр камеры ${camera.name || camera.id}`}
                src={`/api/cameras/${encodeURIComponent(camera.id)}/mjpeg?board_overlay=${showBoard ? 1 : 0}&width=480`}
              />
              <figcaption>
                <span className={`status-dot ${camera.online ? "online" : ""}`} />
                {camera.name || camera.id}
              </figcaption>
            </figure>
          ))}
        </div>
      ) : (
        <div className="empty-previews">Ожидание камер. Откройте ссылку /cam на телефоне.</div>
      )}
    </section>
  );
}
