.PHONY: dashboard summary flops recsys-setup recsys-neumf

help:
	@echo "Available targets:"
	@echo "  dashboard - Start the dashboard server"
	@echo "  summary   - Generate a summary of the model (requires --model argument)"
	@echo "  flops     - Profile the FLOPs for all registered models"

dashboard:
	@echo "Go to http://localhost:8000/docs/experiment_dashboard.html to view the dashboard"
	@python3 dashboard_server.py

summary:
	@python3 tools/model_summary.py --model=$(model)

flops:
	@python3 tools/profile_flops.py