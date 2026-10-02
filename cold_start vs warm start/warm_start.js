import http from 'k6/http';
import { check } from 'k6';
import { Trend } from 'k6/metrics';

const latenza_logistic = new Trend('latenza_logistic');
const latenza_gradient = new Trend('latenza_gradient');
const latenza_random = new Trend('latenza_random');

export default function () {
  const payload = JSON.stringify({
    "instances": [
      [1.5, 2.3, 3.1, 4.0, 5.5, 1.1, 1.1, 1.1, 1.1, 1.1,
       1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1,
       1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1]
    ]
  });

  const params = {
    headers: { 'Content-Type': 'application/json' },
    timeout: '30s'
  };

  let res_log = http.post('http://breast-cancer-logistic-regression-predictor.kserve-inference.svc.cluster.local/v1/models/breast-cancer-logistic-regression:predict', payload, params);
  check(res_log, { 'Logistic status è 200': (r) => r.status === 200 });
  latenza_logistic.add(res_log.timings.duration);

  let res_grad = http.post('http://breast-cancer-gradient-boosting-predictor.kserve-inference.svc.cluster.local/v1/models/breast-cancer-gradient-boosting:predict', payload, params);
  check(res_grad, { 'Gradient status è 200': (r) => r.status === 200 });
  latenza_gradient.add(res_grad.timings.duration);

  let res_rf = http.post('http://breast-cancer-random-forest-predictor.kserve-inference.svc.cluster.local/v1/models/breast-cancer-random-forest:predict', payload, params);
  check(res_rf, { 'Random Forest status è 200': (r) => r.status === 200 });
  latenza_random.add(res_rf.timings.duration);
}

export function handleSummary(data) {
  const l_log = data.metrics.latenza_logistic ? data.metrics.latenza_logistic.values.avg.toFixed(2) : 0;
  const l_grad = data.metrics.latenza_gradient ? data.metrics.latenza_gradient.values.avg.toFixed(2) : 0;
  const l_rf = data.metrics.latenza_random ? data.metrics.latenza_random.values.avg.toFixed(2) : 0;

  console.log(`Logistic Regression: ${l_log} ms`);
  console.log(`Gradient Boosting: ${l_grad} ms`);
  console.log(`Random Forest: ${l_rf} ms`);

  return { 'stdout': '' };
}
