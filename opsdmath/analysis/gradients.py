"""Analytic logit derivatives for the gradient audit."""


def gradients(student, teacher, temperature, clip):
    """Return gradients with respect to student logits, with p for student and q for teacher."""
    p_log = (student.float() / temperature).log_softmax(-1)
    q_log = (teacher.float() / temperature).log_softmax(-1)
    p, q = p_log.exp(), q_log.exp()
    capped = q * (q_log-p_log) > clip
    active = q.masked_fill(capped, 0)
    reverse = p_log-q_log
    return p, {
        "clipped": (p*active.sum(-1, keepdim=True)-active)/temperature,
        "forward": (p-q)/temperature,
        "reverse": p*(reverse-(p*reverse).sum(-1, keepdim=True))/temperature,
    }, (q*capped).sum(-1)


def unit_cos(a, b):
    norm = float(a.norm()*b.norm())
    return float((a*b).sum())/norm if norm else None
