from phy import IPlugin

'''
Since phy lacks native event hooks or refresh triggers, making group_order update in real time requires workarounds that
are not ideal for performance or reliability. The current implementation—where the metric is set up once at 
initialization—works well for sorting clusters in a static way, but it won't reflect updates in real time. 
Achieving true real-time updates would need phy to support a more event-driven or refreshable design.
'''

class GoodLabelsPlugin(IPlugin):
    def attach_to_controller(self, controller):
        """
        Attach the plugin to the controller. Sorts clusters with good first, then mua, then noise.
        """

        def group_order(cluster_id):
            """Return a numeric value for sorting (good=1, mua=2, noise=3)"""
            metadata = controller.model.metadata or {}
            groups = metadata.get('group', {})
            label = groups.get(cluster_id, None)

            order_dict = {
                'good': 1,  # good units first
                'mua': 2,  # then mua
                'noise': 3,  # then noise
                None: 4  # unlabeled last
            }
            return order_dict.get(label, 4)

        # Register the metric
        controller.cluster_metrics['group_order'] = group_order