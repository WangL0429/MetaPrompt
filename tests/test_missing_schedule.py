import unittest

from clip.datamodules.missing_schedule import MissingScheduleSampler, make_schedule


class MissingScheduleTest(unittest.TestCase):
    def test_course_reorders_only_the_prefix(self):
        course = make_schedule(1200, 0, "curriculum")
        random = make_schedule(1200, 0, "random")
        cutoff = 960
        self.assertEqual(course[cutoff:], random[cutoff:])
        self.assertCountEqual(course[:cutoff], random[:cutoff])
        stages = [int(ratio >= 0.4) + int(ratio >= 0.7) for ratio, _, _ in course[:cutoff]]
        self.assertEqual(stages, sorted(stages))
        self.assertEqual(set(stages), {0, 1, 2})
        for stage in {0, 1, 2}:
            self.assertEqual({batch[1] for batch, s in zip(course, stages) if s == stage}, {"text", "image", "both"})

    def test_epoch_sampler_delivers_planned_counts_and_same_data_order(self):
        dataset = range(165)
        options = dict(batch_size=20, epochs=3, seed=0, num_replicas=1, rank=0)
        course = MissingScheduleSampler(dataset, order="curriculum", **options)
        random = MissingScheduleSampler(dataset, order="random", **options)
        for epoch in range(3):
            course.set_epoch(epoch)
            random.set_epoch(epoch)
            samples = list(course)
            self.assertEqual([s[0] for s in samples], [s[0] for s in random])
            self.assertEqual(len(samples), len(course))
            for start in range(0, len(samples), 20):
                batch = samples[start:start+20]
                _, _, ratio = batch[0]
                self.assertEqual({s[2] for s in batch}, {ratio})
                self.assertEqual(sum(s[1] != 0 for s in batch), round(20 * ratio))
        with self.assertRaisesRegex(ValueError, "schedule"):
            make_schedule(20, 0, "invalid")


if __name__ == "__main__":
    unittest.main()
