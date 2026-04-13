import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "../api";

export default function TopicComposer({ onCreated }) {
  const [label, setLabel] = useState("");
  const qc = useQueryClient();
  const mutation = useMutation({
    mutationFn: (lbl) => api.createTopic(lbl, ""),
    onSuccess: (created) => {
      qc.invalidateQueries({ queryKey: ["topics"] });
      setLabel("");
      onCreated && onCreated(created);
    },
  });

  const submit = (e) => {
    e.preventDefault();
    const v = label.trim();
    if (!v) return;
    mutation.mutate(v);
  };

  return (
    <form className="composer" onSubmit={submit}>
      <input
        placeholder="add topic… (e.g. central bank liquidity)"
        value={label}
        onChange={(e) => setLabel(e.target.value)}
        disabled={mutation.isPending}
      />
      <button type="submit" disabled={mutation.isPending || !label.trim()}>
        {mutation.isPending ? "…" : "+"}
      </button>
    </form>
  );
}
