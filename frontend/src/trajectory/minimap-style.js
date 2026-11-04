export function minimapStyle(track) {
  if (track.source === "reference") return { color: "#ffc447", dashed: false, width: 5, order: 2 };
  if (track.source === "estimate" || track.live) return { color: "#ff58ba", dashed: true, width: 2.5, order: 4 };
  return { color: track.color || "#a7ed76", dashed: false, width: 3, order: 6 };
}
