export default function PhaseBadge({ phase }) {
  if (!phase) return <div className="skeleton" style={{ height: 80 }} />;
  const momentumArrow = phase.momentum > 0.01 ? "\u2197"
    : phase.momentum < -0.01 ? "\u2198" : "\u2192";
  return (
    <div>
      <div className="phase-badge">
        <span className="emoji">{phase.emoji}</span>
        <span>{phase.label}</span>
      </div>
      <div className="kpi">
        <div className="kpi-item">
          <div className="k">% of peak</div>
          <div className="v">{(phase.pct_of_peak * 100).toFixed(0)}%</div>
        </div>
        <div className="kpi-item">
          <div className="k">momentum</div>
          <div className="v">
            {momentumArrow} {(phase.momentum * 100).toFixed(1)}
          </div>
        </div>
        <div className="kpi-item">
          <div className="k">cycle progress</div>
          <div className="v">
            {phase.cycle && phase.cycle.cycle_progress_pct !== null
              ? phase.cycle.cycle_progress_pct.toFixed(2)
              : "–"}
          </div>
        </div>
      </div>
    </div>
  );
}
