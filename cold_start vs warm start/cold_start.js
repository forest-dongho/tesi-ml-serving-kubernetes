import http from 'k6/http';
import { check } from 'k6';
import { Trend } from 'k6/metrics';

// Metriche personalizzate
const latenza_logistic = new Trend('latenza_logistic');
const latenza_gradient = new Trend('latenza_gradient');
const latenza_random = new Trend('latenza_random');

export default function () {
  const payload = JSON.stringify({
    "instances": [
      [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
    ]
  });

  const params = {
    headers: { 'Content-Type': 'application/json' },
    timeout: '120s' 
  };

  console.log("Inviando richiesta Logistic Regression...");
  let res_log = http.post('http://breast-cancer-logistic-regression-predictor.kserve-inference.svc.cluster.local/v1/models/breast-cancer-logistic-regression:predict', payload, params);
  check(res_log, { 'Logistic status è 200': (r) => r.status === 200 });
  latenza_logistic.add(res_log.timings.duration);

  console.log("Inviando richiesta Gradient Boosting...");
  let res_grad = http.post('http://breast-cancer-gradient-boosting-predictor.kserve-inference.svc.cluster.local/v1/models/breast-cancer-gradient-boosting:predict', payload, params);
  check(res_grad, { 'Gradient status è 200': (r) => r.status === 200 });
  latenza_gradient.add(res_grad.timings.duration);

  console.log("Inviando richiesta Random Forest...");
  let res_rf = http.post('http://breast-cancer-random-forest-predictor.kserve-inference.svc.cluster.local/v1/models/breast-cancer-random-forest:predict', payload, params);
  check(res_rf, { 'Random Forest status è 200': (r) => r.status === 200 });
  latenza_random.add(res_rf.timings.duration);
}

export function handleSummary(data) {
  const l_log = data.metrics.latenza_logistic ? data.metrics.latenza_logistic.values.avg.toFixed(2) : 0;
  const l_grad = data.metrics.latenza_gradient ? data.metrics.latenza_gradient.values.avg.toFixed(2) : 0;
  const l_rf = data.metrics.latenza_random ? data.metrics.latenza_random.values.avg.toFixed(2) : 0;

  const csv_content = `Modello,Latenza_ColdStart_ms\nLogistic_Regression,${l_log}\nGradient_Boosting,${l_grad}\nRandom_Forest,${l_rf}\n`;

  return {
    '/tmp/cold_start.csv': csv_content,
    'stdout': "Test completato! Risultati CSV generati in /tmp/cold_start.csv\n"
  };

